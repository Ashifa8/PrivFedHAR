"""
model.py
========
PrivFedHAR Neural Architecture:
  GRU-LSTM Hybrid Encoder + Multi-Head Attention + Subject Adaptive Layer Normalization (SALN)

Mathematical formulation:
  X_proj  = ReLU(W_p X + b_p)                         -- Eq (1)
  H_LSTM  = SALN_LSTM(LSTM_2(X_proj))                 -- Eq (2)
  A_out   = MHA(H_LSTM, H_LSTM, H_LSTM)               -- Eq (3)
  c_attn  = mean_t(LN(H_LSTM + A_out))                -- Eq (4)
  c_GRU   = SALN_GRU(h_T^GRU)                         -- Eq (5)
  ŷ       = MLP([c_attn ; c_GRU])                     -- Eq (6)
  SALN(h) = γ_s ⊙ LN(h) + β_s                        -- Eq (7)

Authors: Ashifa Ikram, Shanzae Khan, Atif Saeed
         FAST NUCES Islamabad
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubjectAdaptiveLayerNorm(nn.Module):
    """
    Subject Adaptive Layer Normalization (SALN).

    Extends standard LayerNorm with per-subject learnable affine parameters
    (γₛ, βₛ) that remain on the client device and are NEVER transmitted
    to the server during FedAvg aggregation.

    Parameters per client: 2 × hidden_dim = 256  (negligible vs ~471K total)

    SALN(h; γₛ, βₛ) = γₛ ⊙ LayerNorm(h) + βₛ
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.base_ln       = nn.LayerNorm(normalized_shape, eps=eps,
                                          elementwise_affine=True)
        # Subject-specific parameters — initialized to identity transform
        self.subject_gamma = nn.Parameter(torch.ones(normalized_shape))
        self.subject_beta  = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.subject_gamma * self.base_ln(x) + self.subject_beta


class TemporalAttention(nn.Module):
    """
    Multi-Head Self-Attention over LSTM output sequence.

    MHA(Q,K,V) = Concat(head₁,...,headₙ) W^O
    headᵢ = Softmax(QW_i^Q (KW_i^K)^T / √dₖ) V W_i^V

    with Q = K = V = H_LSTM,  dₖ = hidden_dim / n_heads = 32
    Attention weights W_attn ∈ ℝ^(T×T) emphasize discriminative segments.
    """
    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, T, hidden_dim)  — LSTM output sequence
        Returns:
            context:      (batch, hidden_dim)  — mean-pooled attention output
            attn_weights: (batch, T, T)        — attention weight matrix
        """
        attn_out, attn_weights = self.attn(x, x, x)
        out     = self.norm(x + attn_out)          # residual + LN
        context = out.mean(dim=1)                   # temporal mean pooling
        return context, attn_weights


class PersonalizedFedLSTMGRU(nn.Module):
    """
    PrivFedHAR: Subject-Adaptive GRU-LSTM Attention Network (SAGLAN)

    Architecture:
        Input → Linear Projection → 2-layer LSTM → SALN → MHA
                                 → 1-layer GRU  → SALN
                               Concat([c_attn ; c_gru]) → MLP → logits

    Key design:
        - SALN parameters (subject_gamma, subject_beta) are CLIENT-LOCAL
          and excluded from FedAvg aggregation via get_shared_params()
        - Total params: ~471K   |   SALN params: 256 per client
    """
    def __init__(self,
                 input_dim:   int,
                 hidden_dim:  int,
                 num_classes: int,
                 subject_id:  int   = 0,
                 n_layers:    int   = 2,
                 dropout:     float = 0.4,
                 n_heads:     int   = 4):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.subject_id  = subject_id
        self.num_classes = num_classes

        # Input projection: F → H
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 2-layer LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False
        )

        # SALN after LSTM  (subject-specific — NOT aggregated)
        self.sa_ln_lstm = SubjectAdaptiveLayerNorm(hidden_dim)

        # Multi-Head Attention (4 heads, dₖ=32)
        self.temporal_attn = TemporalAttention(hidden_dim,
                                               n_heads=n_heads,
                                               dropout=dropout)

        # 1-layer GRU encoder (captures short-range local dynamics)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # SALN after GRU  (subject-specific — NOT aggregated)
        self.sa_ln_gru = SubjectAdaptiveLayerNorm(hidden_dim)

        # Classification head: 2H → H → C
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, T, F)  raw sensor windows
        Returns:
            logits:       (batch, num_classes)
            attn_weights: (batch, T, T)
        """
        # Input projection  (F → H)
        x_proj = self.input_proj(x)                        # (B, T, H)

        # LSTM branch
        lstm_out, _ = self.lstm(x_proj)                    # (B, T, H)
        lstm_out    = self.sa_ln_lstm(lstm_out)             # SALN (local)
        c_attn, attn_w = self.temporal_attn(lstm_out)      # (B, H)

        # GRU branch
        gru_out, gru_h = self.gru(lstm_out)                # (B, T, H), (1,B,H)
        gru_out = self.sa_ln_gru(gru_out)
        c_gru   = gru_h.squeeze(0)                         # (B, H)

        # Dual-branch fusion
        combined = torch.cat([c_attn, c_gru], dim=-1)      # (B, 2H)
        logits   = self.classifier(combined)                # (B, C)

        return logits, attn_w

    # ── Federated parameter management ─────────────────────────────────────
    def get_shared_params(self) -> dict:
        """Return shared backbone parameters (transmitted to server)."""
        return {
            k: v for k, v in self.state_dict().items()
            if 'subject_gamma' not in k and 'subject_beta' not in k
        }

    def get_personal_params(self) -> dict:
        """Return SALN parameters (client-local, never transmitted)."""
        return {
            k: v for k, v in self.state_dict().items()
            if 'subject_gamma' in k or 'subject_beta' in k
        }

    def count_params(self) -> dict:
        """Return parameter counts for shared vs personal."""
        total    = sum(p.numel() for p in self.parameters())
        personal = sum(p.numel() for k, p in self.named_parameters()
                       if 'subject_gamma' in k or 'subject_beta' in k)
        return {'total': total, 'shared': total - personal, 'personal': personal}
