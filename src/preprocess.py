"""
preprocess.py
=============
PAMAP2 raw .dat file loading, sliding-window segmentation,
class-conditional augmentation, and per-fold normalization.

Authors: Ashifa Ikram, Shanzae Khan, Atif Saeed
         FAST NUCES Islamabad
"""

import os
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─── Column schema ───────────────────────────────────────────────────────────
COLUMNS = ['timestamp', 'activityID', 'heart_rate'] + \
          [f'IMU_hand_{i}'  for i in range(17)] + \
          [f'IMU_chest_{i}' for i in range(17)] + \
          [f'IMU_ankle_{i}' for i in range(17)]

VALID_ACTIVITIES = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 24]

ACTIVITY_NAMES = {
    1:'lying',        2:'sitting',       3:'standing',
    4:'walking',      5:'running',       6:'cycling',
    7:'nordic_walk',  9:'watching_TV',   10:'computer',
    11:'car_driving', 12:'asc_stairs',   13:'desc_stairs',
    16:'vacuum',      17:'ironing',      24:'rope_jumping'
}

# 40 features: 9 IMU channels × 3 locations + 1 HR + 12 quaternion channels
FEATURE_COLS = [
    'IMU_hand_0',  'IMU_hand_1',  'IMU_hand_2',
    'IMU_hand_3',  'IMU_hand_4',  'IMU_hand_5',
    'IMU_hand_6',  'IMU_hand_7',  'IMU_hand_8',
    'IMU_chest_0', 'IMU_chest_1', 'IMU_chest_2',
    'IMU_chest_3', 'IMU_chest_4', 'IMU_chest_5',
    'IMU_chest_6', 'IMU_chest_7', 'IMU_chest_8',
    'IMU_ankle_0', 'IMU_ankle_1', 'IMU_ankle_2',
    'IMU_ankle_3', 'IMU_ankle_4', 'IMU_ankle_5',
    'IMU_ankle_6', 'IMU_ankle_7', 'IMU_ankle_8',
    'heart_rate',
    'IMU_hand_13',  'IMU_hand_14',  'IMU_hand_15',  'IMU_hand_16',
    'IMU_chest_13', 'IMU_chest_14', 'IMU_chest_15', 'IMU_chest_16',
    'IMU_ankle_13', 'IMU_ankle_14', 'IMU_ankle_15', 'IMU_ankle_16',
]

WINDOW_SIZE  = 128   # timesteps per window (1.28 s at 100 Hz)
STRIDE       = 64    # 50% overlap
NUM_FEATURES = len(FEATURE_COLS)   # 40
SEED         = 42


# ─── 1. File discovery ────────────────────────────────────────────────────────
def find_dat_files(root_dir: str) -> list[str]:
    """Recursively find all subject*.dat files under root_dir."""
    dat_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.startswith('subject') and f.endswith('.dat'):
                dat_files.append(os.path.join(root, f))
    if not dat_files:
        # Try zip files
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.endswith('.zip'):
                    zip_path = os.path.join(root, f)
                    extract_path = os.path.join(root_dir, 'extracted')
                    os.makedirs(extract_path, exist_ok=True)
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        z.extractall(extract_path)
                    return find_dat_files(extract_path)
    return list(set(dat_files))


# ─── 2. Single-subject loader ─────────────────────────────────────────────────
def load_subject(filepath: str) -> pd.DataFrame | None:
    """Load a single subject .dat file into a DataFrame."""
    try:
        df = pd.read_csv(filepath, sep=r'\s+', header=None)
        if len(df.columns) >= len(COLUMNS):
            df = df.iloc[:, :len(COLUMNS)]
            df.columns = COLUMNS
        else:
            return None
        return df
    except Exception as e:
        print(f"  [WARN] Failed to load {filepath}: {e}")
        return None


# ─── 3. Sliding-window segmentation ──────────────────────────────────────────
def sliding_window(df: pd.DataFrame,
                   window_size: int = WINDOW_SIZE,
                   stride: int = STRIDE) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply sliding window to a subject DataFrame.

    Returns:
        X: (n_windows, window_size, n_features)  float32
        y: (n_windows,)  int64  — class index 0..14
    """
    df = df[df['activityID'].isin(VALID_ACTIVITIES)].dropna(
        subset=FEATURE_COLS + ['activityID'])

    # Build class-index mapping
    act_to_idx = {a: i for i, a in enumerate(VALID_ACTIVITIES)}

    data   = df[FEATURE_COLS].values.astype(np.float32)
    labels = df['activityID'].values

    X_windows, y_windows = [], []
    n = len(data)
    for start in range(0, n - window_size + 1, stride):
        end   = start + window_size
        win_y = labels[start:end]
        # Majority-vote label; skip windows with mixed activities
        unique, counts = np.unique(win_y, return_counts=True)
        majority = unique[np.argmax(counts)]
        if np.max(counts) / window_size >= 0.7:   # ≥ 70% same activity
            X_windows.append(data[start:end])
            y_windows.append(act_to_idx[majority])

    if not X_windows:
        return np.empty((0, window_size, NUM_FEATURES), dtype=np.float32), \
               np.empty(0, dtype=np.int64)

    return (np.array(X_windows, dtype=np.float32),
            np.array(y_windows, dtype=np.int64))


# ─── 4. Class-conditional augmentation ───────────────────────────────────────
def augment_subject(X: np.ndarray, y: np.ndarray,
                    target_min: int = 300,
                    noise_std: float = 0.02,
                    jitter_std: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment under-represented classes with Gaussian noise + temporal jitter.
    Never duplicates exact samples; expands dataset from ~32k → ~60k windows.

    Args:
        X: (n, T, F)  original windows
        y: (n,)       class indices
        target_min:   minimum samples per class after augmentation
        noise_std:    std of additive Gaussian noise  (σ = 0.02)
        jitter_std:   std of temporal shift           (σ = 0.01)
    """
    X_aug, y_aug = list(X), list(y)
    class_counts = Counter(y.tolist())

    for cls, cnt in class_counts.items():
        if cnt < target_min:
            idxs   = np.where(y == cls)[0]
            needed = target_min - cnt
            chosen = np.random.choice(idxs, needed, replace=True)
            for idx in chosen:
                # Gaussian noise augmentation
                noisy = X[idx] + np.random.normal(
                    0, noise_std, X[idx].shape).astype(np.float32)
                # Temporal jitter (circular roll)
                shift = np.random.randint(-5, 5)
                noisy = np.roll(noisy, shift, axis=0)
                X_aug.append(noisy)
                y_aug.append(cls)

    return np.array(X_aug, dtype=np.float32), np.array(y_aug, dtype=np.int64)


# ─── 5. Full pipeline ─────────────────────────────────────────────────────────
def load_and_preprocess(data_dir: str,
                        augment: bool = True,
                        augment_target: int = 300,
                        verbose: bool = True) -> dict[int, tuple]:
    """
    Full preprocessing pipeline:
      1. Find all subject .dat files
      2. Load each subject
      3. Sliding-window segmentation
      4. Class-conditional augmentation (optional)

    Args:
        data_dir:        directory containing subject*.dat files
        augment:         whether to apply augmentation
        augment_target:  minimum windows per class after augmentation
        verbose:         print progress

    Returns:
        encoded_raw: {subject_id: (X, y)}
            X: (n_windows, WINDOW_SIZE, NUM_FEATURES)
            y: (n_windows,)  class indices 0..14
    """
    dat_files = find_dat_files(data_dir)
    if not dat_files:
        raise FileNotFoundError(
            f"No subject*.dat files found under {data_dir}.\n"
            "Download PAMAP2 from https://archive.ics.uci.edu/ml/datasets/"
            "pamap2+physical+activity+monitoring")

    if verbose:
        print(f"Found {len(dat_files)} .dat files")

    encoded_raw: dict[int, tuple] = {}

    for fpath in sorted(dat_files):
        basename = os.path.basename(fpath).lower()
        # Extract subject number from filename
        for sid in range(1, 10):
            if f'subject{sid:01d}' in basename or \
               f'subject10{sid}' in basename or \
               f'subject{100 + sid}' in basename:
                df = load_subject(fpath)
                if df is None:
                    break
                X, y = sliding_window(df)
                if len(X) == 0:
                    break
                if augment:
                    X, y = augment_subject(X, y, target_min=augment_target)
                encoded_raw[sid] = (X, y)
                if verbose:
                    print(f"  Subject {sid}: {len(X):,} windows "
                          f"| Classes: {len(set(y.tolist()))}")
                break

    if verbose:
        total = sum(len(v[0]) for v in encoded_raw.values())
        print(f"\nTotal subjects: {len(encoded_raw)} | "
              f"Total windows: {total:,}")

    return encoded_raw


# ─── 6. LOSO split with per-fold normalization ───────────────────────────────
def loso_split(encoded_raw: dict,
               test_subject: int,
               val_fraction: float = 0.15,
               seed: int = SEED):
    """
    Leave-One-Subject-Out split with leakage-free normalization.

    StandardScaler is fit ONLY on training subjects' windows,
    then applied to validation and test windows.

    Returns:
        client_data_norm: {sid: (X_train, y_train, X_val, y_val)}
        X_test_norm:      (n, T, F) normalized test windows
        y_test:           (n,) test labels
        scaler:           fitted StandardScaler
    """
    all_sids   = sorted(encoded_raw.keys())
    train_sids = [s for s in all_sids if s != test_subject]
    client_data: dict = {}

    for sid in train_sids:
        X, y = encoded_raw[sid]
        if len(X) < 50:
            client_data[sid] = (X, y, X, y)
            continue
        try:
            X_tr, X_v, y_tr, y_v = train_test_split(
                X, y, test_size=val_fraction,
                stratify=y, random_state=seed)
        except ValueError:
            X_tr, X_v, y_tr, y_v = train_test_split(
                X, y, test_size=val_fraction, random_state=seed)
        client_data[sid] = (X_tr, y_tr, X_v, y_v)

    # Fit scaler on training data only
    X_all = np.concatenate([v[0] for v in client_data.values()], axis=0)
    N, T, F = X_all.shape
    scaler = StandardScaler()
    scaler.fit(X_all.reshape(-1, F))

    def norm(X: np.ndarray) -> np.ndarray:
        n, t, f = X.shape
        return scaler.transform(
            X.reshape(-1, f)).reshape(n, t, f).astype(np.float32)

    client_data_norm = {
        sid: (norm(Xtr), ytr, norm(Xv), yv)
        for sid, (Xtr, ytr, Xv, yv) in client_data.items()
    }

    X_test, y_test = encoded_raw[test_subject]
    return client_data_norm, norm(X_test), y_test, scaler
