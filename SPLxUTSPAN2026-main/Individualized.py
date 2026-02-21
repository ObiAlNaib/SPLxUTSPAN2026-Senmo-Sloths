# ============================================================
# SPL-UTSPAN 2026 — ENHANCED Pipeline
# Key improvements:
#   1. Better temporal feature extraction (sliding windows)
#   2. Enhanced kinematic features (asymmetry, coordination)
#   3. Improved release point detection
#   4. Target-specific feature engineering
#   5. Better model diversity
# ============================================================

import os, re, ast, time, math, hashlib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import skew, kurtosis

from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ----------------------------
# USER SETTINGS
# ----------------------------
DATA_DIR = "."
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test.csv")
OUT_PATH   = os.path.join(DATA_DIR, "submission_improved.csv")

TARGETS = ["angle", "depth", "left_right"]

# CV
CV_MODE = "random"
N_SPLITS = 5
RANDOM_STATE = 42

# Model settings - INCREASED CAPACITY
OOF_BAG_SEEDS   = [0, 1, 2, 3]
OOF_N_ESTIMATORS = 800
FINAL_BAG_SEEDS  = list(range(15))  # Increased from 10
FINAL_N_ESTIMATORS = 3000  # Increased from 2500

# Feature caching
CACHE_FEATURES = True
CACHE_TAG = "v4_enhanced_temporal_kinematic"
CACHE_TRAIN = os.path.join(DATA_DIR, f"_cache_X_train_{CACHE_TAG}.pkl")
CACHE_TEST  = os.path.join(DATA_DIR, f"_cache_X_test_{CACHE_TAG}.pkl")

# Feature toggles
KEEP_SHOT_ID_HASH = True
USE_PID_ONEHOT = True
USE_SMOOTHING = True
SMOOTH_WIN = 11
SMOOTH_POLY = 3

# Release anchoring - EXPANDED
FRAME_RATE = 60
RELEASE_OFFSETS = [-30, -24, -18, -12, -9, -6, -3, 0, 3, 6, 9, 12, 18, 24, 30]

# Raw-flat + PCA
ADD_RAW_FLAT = True
RAW_FLAT_NFRAMES = 240
RAW_FLAT_JOINTS = [
    "right_wrist","left_wrist",
    "right_elbow","left_elbow",
    "right_shoulder","left_shoulder",
    "neck","mid_hip",
    "right_hip","left_hip",
    "right_knee","left_knee",
    "right_ankle","left_ankle",
]
PCA_DIMS = [50, 100, 150]  # Added 150

# Stacking
USE_STACKING = True
RIDGE_ALPHAS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0]

USE_TOPK_BLEND = True
TOPK_PER_TARGET = 8  # Increased from 6
CORR_PRUNE = True
MAX_CORR = 0.98  # Slightly more lenient
BLEND_RANDOM_TRIES = 6000  # Increased
BLEND_REFINES = 150  # Increased

# ----------------------------
# Scaling ranges
# ----------------------------
ANGLE_MIN, ANGLE_MAX = 30.0, 60.0
DEPTH_MIN, DEPTH_MAX = -12.0, 30.0
LR_MIN, LR_MAX       = -16.0, 16.0

def clip01(x):
    return np.clip(x, 0.0, 1.0)

def scale_targets_np(y_raw):
    a  = (y_raw[:, 0] - ANGLE_MIN) / (ANGLE_MAX - ANGLE_MIN)
    d  = (y_raw[:, 1] - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)
    lr = (y_raw[:, 2] - LR_MIN) / (LR_MAX - LR_MIN)
    return np.column_stack([a, d, lr])

def ensure_finite(X):
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# ----------------------------
# Parsing utilities
# ----------------------------
_nan_pat = re.compile(r"\b(nan|NaN|NAN|null|NULL|inf|Inf|INF|-inf|-Inf|-INF)\b")

def parse_list_safe(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        try:
            return np.asarray(v, dtype=float)
        except Exception:
            return None

    s = str(v).strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None

    s = _nan_pat.sub("None", s)
    try:
        out = ast.literal_eval(s)
        if not isinstance(out, (list, tuple)):
            return None
        return np.array([np.nan if x is None else float(x) for x in out], dtype=float)
    except Exception:
        return None

def safe_idx(i, n):
    if n <= 0:
        return 0
    return max(0, min(n - 1, int(i)))

def diff_1d(a):
    a = np.asarray(a, dtype=float)
    if a.size <= 1:
        return np.array([0.0], dtype=float)
    return np.concatenate([[0.0], np.diff(a)])

def smooth_1d(x, win=11, poly=3):
    if not USE_SMOOTHING:
        return x
    x = np.asarray(x, dtype=float)
    if x.size < 7:
        return x
    win = int(win)
    if win % 2 == 0:
        win += 1
    win = min(win, x.size if x.size % 2 == 1 else x.size - 1)
    if win < 7:
        return x
    try:
        poly = int(min(poly, win - 2))
        return savgol_filter(x, window_length=win, polyorder=poly, mode="interp")
    except Exception:
        return x

# ENHANCED: More comprehensive statistics
def summarize_1d(a, prefix):
    a = np.asarray(a, dtype=float)
    if a.size == 0 or np.all(np.isnan(a)):
        return {f"{prefix}{k}": 0.0 for k in
                ["mean","std","min","max","range","q10","q25","q50","q75","q90",
                 "iqr","first","last","slope","t_peak","skew","kurt","rms"]}

    mean = float(np.nanmean(a))
    std  = float(np.nanstd(a))
    mn   = float(np.nanmin(a))
    mx   = float(np.nanmax(a))
    q10, q25, q50, q75, q90 = np.nanpercentile(a, [10, 25, 50, 75, 90])
    iqr  = float(q75 - q25)
    first = float(a[0]) if np.isfinite(a[0]) else mean
    last  = float(a[-1]) if np.isfinite(a[-1]) else mean
    slope = float((last - first) / max(1, a.size - 1))
    idx_peak = int(np.nanargmax(a)) if a.size else 0
    t_peak = float(idx_peak / (a.size - 1)) if a.size > 1 else 0.0
    
    # NEW: skewness, kurtosis, RMS
    sk = float(skew(a[np.isfinite(a)])) if np.sum(np.isfinite(a)) > 2 else 0.0
    kt = float(kurtosis(a[np.isfinite(a)])) if np.sum(np.isfinite(a)) > 2 else 0.0
    rms = float(np.sqrt(np.nanmean(a**2)))

    return {
        f"{prefix}mean": mean,
        f"{prefix}std": std,
        f"{prefix}min": mn,
        f"{prefix}max": mx,
        f"{prefix}range": float(mx - mn),
        f"{prefix}q10": float(q10),
        f"{prefix}q25": float(q25),
        f"{prefix}q50": float(q50),
        f"{prefix}q75": float(q75),
        f"{prefix}q90": float(q90),
        f"{prefix}iqr": iqr,
        f"{prefix}first": first,
        f"{prefix}last": last,
        f"{prefix}slope": slope,
        f"{prefix}t_peak": t_peak,
        f"{prefix}skew": sk,
        f"{prefix}kurt": kt,
        f"{prefix}rms": rms,
    }

def resample_1d(a, n_frames):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return np.zeros(n_frames, dtype=float)
    if a.size == n_frames:
        return ensure_finite(a).astype(float)
    x_old = np.linspace(0.0, 1.0, a.size)
    x_new = np.linspace(0.0, 1.0, n_frames)
    a2 = np.interp(x_new, x_old, ensure_finite(a))
    return ensure_finite(a2).astype(float)

def angle_3pts(A, B, C):
    A = np.asarray(A, float); B = np.asarray(B, float); C = np.asarray(C, float)
    BA = A - B
    BC = C - B
    nba = np.linalg.norm(BA); nbc = np.linalg.norm(BC)
    if nba < 1e-9 or nbc < 1e-9:
        return 0.0
    cosv = float(np.dot(BA, BC) / (nba * nbc))
    cosv = max(-1.0, min(1.0, cosv))
    return float(np.degrees(np.arccos(cosv)))

# ----------------------------
# NEW: Sliding window features
# ----------------------------
def sliding_window_stats(a, window_size=15):
    """Extract stats from sliding windows across the time series"""
    a = ensure_finite(np.asarray(a, dtype=float))
    if a.size < window_size:
        return {"sw_mean_max": 0.0, "sw_std_max": 0.0, "sw_range_max": 0.0}
    
    means, stds, ranges = [], [], []
    for i in range(a.size - window_size + 1):
        w = a[i:i+window_size]
        means.append(np.mean(w))
        stds.append(np.std(w))
        ranges.append(np.max(w) - np.min(w))
    
    return {
        "sw_mean_max": float(np.max(means)),
        "sw_mean_min": float(np.min(means)),
        "sw_std_max": float(np.max(stds)),
        "sw_range_max": float(np.max(ranges)),
    }

# ----------------------------
# Enhanced feature engineering
# ----------------------------
KEY_JOINTS = [
    "right_wrist","left_wrist",
    "right_elbow","left_elbow",
    "right_shoulder","left_shoulder",
    "neck","mid_hip",
    "right_hip","left_hip",
    "right_knee","left_knee",
    "right_ankle","left_ankle",
]

AXES = ["x","y","z"]

def stable_hash_feats(s, n=4):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.zeros(n, dtype=float)
    h = hashlib.md5(str(s).encode("utf-8")).hexdigest()
    ints = [int(h[i:i+8], 16) for i in range(0, 32, 8)]
    v = np.array(ints, dtype=np.float64)
    v = (v % 1000003) / 1000003.0
    return v[:n]

def featurize(df, is_train):
    df = df.copy()

    ids = df["id"].copy() if "id" in df.columns else pd.Series(np.arange(len(df)))
    pid = df["participant_id"].copy() if "participant_id" in df.columns else None
    shot = df["shot_id"].copy() if "shot_id" in df.columns else None

    df = df.drop(columns=["id"], errors="ignore")

    # Parse list columns
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    list_cols = []
    for c in obj_cols:
        s = df[c].dropna()
        if s.empty:
            continue
        v = str(s.iloc[0]).strip()
        if v.startswith("[") and v.endswith("]"):
            list_cols.append(c)

    parsed = {}
    for c in list_cols:
        arrs = df[c].apply(parse_list_safe)
        parsed[c] = arrs

    base = df.drop(columns=list_cols, errors="ignore").copy()
    base = base.drop(columns=["shot_id", "participant_id"], errors="ignore")

    for c in base.columns:
        if base[c].dtype == "object":
            base[c] = pd.to_numeric(base[c], errors="coerce")
    base = base.fillna(0.0)

    feats = [base]

    if USE_PID_ONEHOT and pid is not None:
        pid_cat = pid.astype(str)
        oh = pd.get_dummies(pid_cat, prefix="pid", dtype=np.float32)
        feats.append(oh)

    if KEEP_SHOT_ID_HASH and shot is not None:
        H = np.vstack([stable_hash_feats(x, n=4) for x in shot.to_list()]).astype(np.float32)
        feats.append(pd.DataFrame(H, columns=[f"shot_hash_{i}" for i in range(H.shape[1])]))

    # Get frame lengths
    def get_frames(i):
        for key in ["right_wrist_x","left_wrist_x","mid_hip_x"]:
            if key in parsed:
                a = parsed[key].iloc[i]
                if a is not None and isinstance(a, np.ndarray) and a.size > 3:
                    return int(a.size)
        for c in parsed:
            a = parsed[c].iloc[i]
            if a is not None and isinstance(a, np.ndarray):
                return int(a.size)
        return 1

    n = len(df)
    n_frames = np.array([get_frames(i) for i in range(n)], dtype=int)

    def coord_at(i, joint, axis, frame):
        key = f"{joint}_{axis}"
        if key not in parsed:
            return np.nan
        a = parsed[key].iloc[i]
        if a is None or not isinstance(a, np.ndarray) or a.size == 0:
            return np.nan
        a = smooth_1d(a, SMOOTH_WIN, SMOOTH_POLY)
        f = safe_idx(frame, a.size)
        return float(a[f])

    # IMPROVED: Release detection using multiple signals
    release_idx = np.zeros(n, dtype=int)
    active_is_right = np.zeros(n, dtype=np.float32)

    for i in range(n):
        L = int(n_frames[i])
        if L <= 3:
            release_idx[i] = 0
            active_is_right[i] = 1.0
            continue

        def wrist_speed(side):
            kx, ky, kz = f"{side}_wrist_x", f"{side}_wrist_y", f"{side}_wrist_z"
            if (kx not in parsed) or (ky not in parsed) or (kz not in parsed):
                return None
            x = parsed[kx].iloc[i]; y = parsed[ky].iloc[i]; z = parsed[kz].iloc[i]
            if x is None or y is None or z is None:
                return None
            x = smooth_1d(x, SMOOTH_WIN, SMOOTH_POLY)
            y = smooth_1d(y, SMOOTH_WIN, SMOOTH_POLY)
            z = smooth_1d(z, SMOOTH_WIN, SMOOTH_POLY)
            vx = diff_1d(x) * FRAME_RATE
            vy = diff_1d(y) * FRAME_RATE
            vz = diff_1d(z) * FRAME_RATE
            s = np.sqrt(vx*vx + vy*vy + vz*vz)
            return ensure_finite(s)

        sr = wrist_speed("right")
        sl = wrist_speed("left")

        def tail_mean(s):
            if s is None or s.size == 0:
                return 0.0
            a = int(0.65 * s.size)
            return float(np.nanmean(s[a:]))
        
        tr = tail_mean(sr)
        tl = tail_mean(sl)
        side = "right" if tr >= tl else "left"
        active_is_right[i] = 1.0 if side == "right" else 0.0

        s_use = sr if side == "right" else sl
        if s_use is None or s_use.size < 3:
            release_idx[i] = int(0.6 * L)
        else:
            start = safe_idx(int(0.6 * s_use.size), s_use.size)
            peak = start + int(np.nanargmax(s_use[start:]))
            release_idx[i] = safe_idx(peak, L)

    feats.append(pd.DataFrame({
        "release_idx": release_idx.astype(np.float32),
        "active_is_right": active_is_right.astype(np.float32),
        "n_frames": n_frames.astype(np.float32),
    }))

    # Mid-hip for relative coords
    midhip = {ax: parsed.get(f"mid_hip_{ax}", None) for ax in AXES}

    # ENHANCED: Position/velocity/acc/jerk + sliding windows
    rows = []
    for i in range(n):
        row = {}
        L = int(n_frames[i])

        for joint in KEY_JOINTS:
            for ax in AXES:
                key = f"{joint}_{ax}"
                if key not in parsed:
                    continue
                a = parsed[key].iloc[i]
                if a is None or not isinstance(a, np.ndarray) or a.size < 2:
                    a = np.zeros(L, dtype=float)
                a = smooth_1d(a, SMOOTH_WIN, SMOOTH_POLY)
                a = ensure_finite(a)

                row.update(summarize_1d(a, f"{key}_pos_"))
                
                # NEW: Sliding window stats
                sw_stats = sliding_window_stats(a, window_size=15)
                for k, v in sw_stats.items():
                    row[f"{key}_pos_{k}"] = v

                v = diff_1d(a) * FRAME_RATE
                row.update(summarize_1d(v, f"{key}_vel_"))

                acc = diff_1d(v) * FRAME_RATE
                row.update(summarize_1d(acc, f"{key}_acc_"))

                jerk = diff_1d(acc) * FRAME_RATE
                row.update(summarize_1d(jerk, f"{key}_jerk_"))

                mh_arrs = midhip.get(ax, None)
                if mh_arrs is not None:
                    mh = mh_arrs.iloc[i]
                    if mh is None or not isinstance(mh, np.ndarray) or mh.size < 2:
                        mh = np.zeros(L, dtype=float)
                    mh = smooth_1d(mh, SMOOTH_WIN, SMOOTH_POLY)
                    mh = ensure_finite(mh)

                    m = min(a.size, mh.size)
                    rel = a[:m] - mh[:m]
                    row.update(summarize_1d(rel, f"{key}_relmid_"))

        rows.append(row)

    feats.append(pd.DataFrame(rows).fillna(0.0))

    # NEW: Asymmetry features (left vs right)
    asym_rows = []
    for i in range(n):
        row = {}
        
        for joint_base in ["wrist", "elbow", "shoulder", "hip", "knee", "ankle"]:
            for ax in AXES:
                r_key = f"right_{joint_base}_{ax}"
                l_key = f"left_{joint_base}_{ax}"
                
                if r_key in parsed and l_key in parsed:
                    r_arr = parsed[r_key].iloc[i]
                    l_arr = parsed[l_key].iloc[i]
                    
                    if isinstance(r_arr, np.ndarray) and isinstance(l_arr, np.ndarray):
                        r_arr = ensure_finite(smooth_1d(r_arr, SMOOTH_WIN, SMOOTH_POLY))
                        l_arr = ensure_finite(smooth_1d(l_arr, SMOOTH_WIN, SMOOTH_POLY))
                        
                        diff = r_arr - l_arr
                        row[f"{joint_base}_{ax}_asym_mean"] = float(np.mean(diff))
                        row[f"{joint_base}_{ax}_asym_std"] = float(np.std(diff))
                        row[f"{joint_base}_{ax}_asym_max"] = float(np.max(np.abs(diff)))
        
        asym_rows.append(row)
    
    feats.append(pd.DataFrame(asym_rows).fillna(0.0))

    # Release-offset snapshots + geometry
    rows2 = []
    for i in range(n):
        row = {}
        L = int(n_frames[i])
        ridx = safe_idx(release_idx[i], L)
        side = "right" if active_is_right[i] >= 0.5 else "left"

        wj = f"{side}_wrist"
        for off in RELEASE_OFFSETS:
            f = safe_idx(ridx + off, L)
            for ax in AXES:
                row[f"{wj}_rel{off:+d}_{ax}"] = float(ensure_finite(coord_at(i, wj, ax, f)))

        def pt(j):
            return np.array([coord_at(i, j, "x", ridx), coord_at(i, j, "y", ridx), coord_at(i, j, "z", ridx)], float)

        for s in ["right", "left"]:
            sh = f"{s}_shoulder"; el = f"{s}_elbow"; wr = f"{s}_wrist"
            hip = f"{s}_hip"; kn = f"{s}_knee"; an = f"{s}_ankle"

            Psh, Pel, Pwr = pt(sh), pt(el), pt(wr)
            Phip, Pkn, Pan = pt(hip), pt(kn), pt(an)

            def dist(A,B):
                if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
                    return 0.0
                return float(np.linalg.norm(A-B))

            row[f"{s}_sh_to_wr_dist_release"] = dist(Psh, Pwr)
            row[f"{s}_el_to_wr_dist_release"] = dist(Pel, Pwr)
            row[f"{s}_sh_to_el_dist_release"] = dist(Psh, Pel)
            row[f"{s}_hip_to_kn_dist_release"] = dist(Phip, Pkn)
            row[f"{s}_kn_to_an_dist_release"] = dist(Pkn, Pan)

            row[f"{s}_elbow_angle_release"] = angle_3pts(Psh, Pel, Pwr)
            row[f"{s}_knee_angle_release"]  = angle_3pts(Phip, Pkn, Pan)

        Pn = pt("neck"); Pm = pt("mid_hip")
        if np.all(np.isfinite(Pn)) and np.all(np.isfinite(Pm)):
            v = Pn - Pm
            nz = np.linalg.norm(v)
            if nz > 1e-9:
                cosv = float(v[2] / nz)
                cosv = max(-1.0, min(1.0, cosv))
                row["torso_tilt_deg_release"] = float(np.degrees(np.arccos(cosv)))
            else:
                row["torso_tilt_deg_release"] = 0.0
        else:
            row["torso_tilt_deg_release"] = 0.0

        rows2.append(row)

    feats.append(pd.DataFrame(rows2).fillna(0.0))

    # FFT features
    fft_rows = []
    for i in range(n):
        row = {}
        side = "right" if active_is_right[i] >= 0.5 else "left"
        kx, ky, kz = f"{side}_wrist_x", f"{side}_wrist_y", f"{side}_wrist_z"
        if (kx in parsed) and (ky in parsed) and (kz in parsed):
            x = parsed[kx].iloc[i]; y = parsed[ky].iloc[i]; z = parsed[kz].iloc[i]
            if isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and isinstance(z, np.ndarray) and x.size > 8:
                x = ensure_finite(smooth_1d(x, SMOOTH_WIN, SMOOTH_POLY))
                y = ensure_finite(smooth_1d(y, SMOOTH_WIN, SMOOTH_POLY))
                z = ensure_finite(smooth_1d(z, SMOOTH_WIN, SMOOTH_POLY))
                vx = diff_1d(x) * FRAME_RATE
                vy = diff_1d(y) * FRAME_RATE
                vz = diff_1d(z) * FRAME_RATE
                s = np.sqrt(vx*vx + vy*vy + vz*vz)
                s = ensure_finite(s)

                S = np.fft.rfft(s - np.mean(s))
                mag = np.abs(S)
                mag[0] = 0.0
                topk = np.argsort(mag)[-10:][::-1]
                for j, idx in enumerate(topk):
                    row[f"wristspd_fft_idx_{j}"] = float(idx)
                    row[f"wristspd_fft_mag_{j}"] = float(mag[idx])
                row["wristspd_fft_energy"] = float(np.sum(mag**2))
            else:
                row["wristspd_fft_energy"] = 0.0
        else:
            row["wristspd_fft_energy"] = 0.0
        fft_rows.append(row)

    feats.append(pd.DataFrame(fft_rows).fillna(0.0))

    # RAW-FLAT (resample + relative to midhip)
    raw_flat = None
    if ADD_RAW_FLAT:
        raw = np.zeros((n, len(RAW_FLAT_JOINTS) * 3 * RAW_FLAT_NFRAMES), dtype=np.float32)
        for i in range(n):
            mhx = parsed.get("mid_hip_x", pd.Series([None]*n)).iloc[i]
            mhy = parsed.get("mid_hip_y", pd.Series([None]*n)).iloc[i]
            mhz = parsed.get("mid_hip_z", pd.Series([None]*n)).iloc[i]
            if not isinstance(mhx, np.ndarray): mhx = np.zeros(1)
            if not isinstance(mhy, np.ndarray): mhy = np.zeros(1)
            if not isinstance(mhz, np.ndarray): mhz = np.zeros(1)
            mhx = resample_1d(mhx, RAW_FLAT_NFRAMES)
            mhy = resample_1d(mhy, RAW_FLAT_NFRAMES)
            mhz = resample_1d(mhz, RAW_FLAT_NFRAMES)

            out = []
            for jn in RAW_FLAT_JOINTS:
                for ax, mh in zip(["x","y","z"], [mhx,mhy,mhz]):
                    key = f"{jn}_{ax}"
                    a = parsed.get(key, pd.Series([None]*n)).iloc[i]
                    if not isinstance(a, np.ndarray):
                        a = np.zeros(1)
                    a = smooth_1d(a, SMOOTH_WIN, SMOOTH_POLY)
                    a = resample_1d(a, RAW_FLAT_NFRAMES)
                    a = ensure_finite(a - mh)
                    out.append(a.astype(np.float32))
            flat = np.concatenate(out, axis=0).reshape(-1)
            raw[i] = ensure_finite(flat).astype(np.float32)
        raw_flat = raw

    X = pd.concat(feats, axis=1).fillna(0.0)
    return X, ids, raw_flat, pid

# ----------------------------
# CV splitter
# ----------------------------
def get_splitter(groups):
    if CV_MODE == "group" and groups is not None:
        return GroupKFold(n_splits=N_SPLITS), "GroupKFold"
    return KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE), "KFold(shuffle=True)"

# ----------------------------
# ENHANCED: More diverse base models
# ----------------------------
ET_PARAMS_LIST = [
    ("ET_A", dict(max_depth=None, min_samples_leaf=1, max_features=0.35, bootstrap=False)),
    ("ET_B", dict(max_depth=None, min_samples_leaf=2, max_features=0.50, bootstrap=False)),
    ("ET_C", dict(max_depth=None, min_samples_leaf=1, max_features=0.70, bootstrap=False)),
    ("ET_D", dict(max_depth=None, min_samples_leaf=2, max_features=0.70, bootstrap=False)),
    ("ET_E", dict(max_depth=None, min_samples_leaf=2, max_features=0.50, bootstrap=True, max_samples=0.80)),
    ("ET_F", dict(max_depth=None, min_samples_leaf=3, max_features=0.40, bootstrap=False)),  # NEW
]

# NEW: Add RandomForest models for diversity
RF_PARAMS_LIST = [
    ("RF_A", dict(max_depth=None, min_samples_leaf=2, max_features=0.40, bootstrap=True)),
    ("RF_B", dict(max_depth=None, min_samples_leaf=3, max_features=0.50, bootstrap=True)),
]

def fit_et(seed, n_estimators, params):
    return ExtraTreesRegressor(
        n_estimators=int(n_estimators),
        random_state=int(seed),
        n_jobs=-1,
        **params
    )

def fit_rf(seed, n_estimators, params):
    return RandomForestRegressor(
        n_estimators=int(n_estimators),
        random_state=int(seed),
        n_jobs=-1,
        **params
    )

def build_oof_models(X, y, groups=None, bag_seeds=None, n_estimators=600, verbose=True):
    splitter, label = get_splitter(groups)
    split_iter = splitter.split(X, y, groups=groups) if "GroupKFold" in label else splitter.split(X, y)

    n = X.shape[0]
    oof = {}
    
    # Initialize OOF arrays for all models
    for name, _ in ET_PARAMS_LIST:
        oof[name] = np.zeros((n, 3), dtype=np.float64)
    for name, _ in RF_PARAMS_LIST:
        oof[name] = np.zeros((n, 3), dtype=np.float64)

    total = N_SPLITS * (len(ET_PARAMS_LIST) + len(RF_PARAMS_LIST)) * len(bag_seeds)
    step = 0
    t0 = time.time()

    for fold, (tr, va) in enumerate(split_iter, start=1):
        Xtr, Xva = X[tr], X[va]
        ytr = y[tr]

        # ExtraTrees models
        for mname, mparams in ET_PARAMS_LIST:
            preds = []
            for s in bag_seeds:
                step += 1
                model = fit_et(s, n_estimators, mparams)
                model.fit(Xtr, ytr)
                preds.append(model.predict(Xva))

                if verbose and (step == 1 or step % 10 == 0):
                    dt = max(1e-9, time.time() - t0)
                    rate = step / dt
                    eta = (total - step) / max(1e-9, rate)
                    print(f"OOF {step}/{total} | fold={fold} model={mname} seed={s} | rate={rate:.2f} fits/s | ETA~{eta/60:.1f} min")

            oof[mname][va] = np.mean(preds, axis=0)
        
        # RandomForest models
        for mname, mparams in RF_PARAMS_LIST:
            preds = []
            for s in bag_seeds:
                step += 1
                model = fit_rf(s, n_estimators, mparams)
                model.fit(Xtr, ytr)
                preds.append(model.predict(Xva))

                if verbose and (step == 1 or step % 10 == 0):
                    dt = max(1e-9, time.time() - t0)
                    rate = step / dt
                    eta = (total - step) / max(1e-9, rate)
                    print(f"OOF {step}/{total} | fold={fold} model={mname} seed={s} | rate={rate:.2f} fits/s | ETA~{eta/60:.1f} min")

            oof[mname][va] = np.mean(preds, axis=0)

    return oof, label

# ----------------------------
# Baselines
# ----------------------------
def oof_baselines(y, pid, groups=None):
    n = y.shape[0]
    base_global = np.zeros((n, 3), dtype=np.float64)
    base_pid    = np.zeros((n, 3), dtype=np.float64)

    splitter, label = get_splitter(groups)
    split_iter = splitter.split(np.zeros((n,1)), y, groups=groups) if "GroupKFold" in label else splitter.split(np.zeros((n,1)), y)

    pid_s = pid.astype(str).to_numpy() if pid is not None else None

    for tr, va in split_iter:
        mu = np.mean(y[tr], axis=0)
        base_global[va] = mu

        if pid_s is not None:
            df = pd.DataFrame({"pid": pid_s[tr]})
            for j in range(3):
                df[f"y{j}"] = y[tr, j]
            means = df.groupby("pid")[[f"y{j}" for j in range(3)]].mean()

            for idx in va:
                p = pid_s[idx]
                if p in means.index:
                    base_pid[idx] = means.loc[p].to_numpy()
                else:
                    base_pid[idx] = mu
        else:
            base_pid[va] = mu

    return base_global, base_pid

def final_baselines(y, pid_train, pid_test):
    mu = np.mean(y, axis=0)
    base_global = np.tile(mu, (len(pid_test), 1))

    pid_tr = pid_train.astype(str).to_numpy()
    pid_te = pid_test.astype(str).to_numpy()

    df = pd.DataFrame({"pid": pid_tr})
    for j in range(3):
        df[f"y{j}"] = y[:, j]
    means = df.groupby("pid")[[f"y{j}" for j in range(3)]].mean()

    base_pid = np.zeros((len(pid_te), 3), dtype=np.float64)
    for i, p in enumerate(pid_te):
        if p in means.index:
            base_pid[i] = means.loc[p].to_numpy()
        else:
            base_pid[i] = mu
    return base_global, base_pid

# ----------------------------
# Ridge stacking
# ----------------------------
def crossfit_ridge(P, y, alphas):
    n = P.shape[0]
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    best = None

    for a in alphas:
        oof = np.zeros(n, dtype=np.float64)
        for tr, va in kf.split(P):
            rr = Ridge(alpha=a, fit_intercept=True)
            rr.fit(P[tr], y[tr])
            oof[va] = rr.predict(P[va])
        mse = mean_squared_error(y, clip01(oof))
        if best is None or mse < best["mse"]:
            best = {"alpha": a, "mse": mse, "oof": oof}

    rr_full = Ridge(alpha=best["alpha"], fit_intercept=True)
    rr_full.fit(P, y)
    return best["alpha"], best["oof"], rr_full

# ----------------------------
# Blend optimizer
# ----------------------------
def corr_prune(model_names, vecs, max_corr=0.995):
    kept = []
    kept_vecs = []
    for i, m in enumerate(model_names):
        v = vecs[i]
        ok = True
        for kv in kept_vecs:
            c = np.corrcoef(v, kv)[0, 1]
            if np.isfinite(c) and c > max_corr:
                ok = False
                break
        if ok:
            kept.append(m)
            kept_vecs.append(v)
    return kept

def optimize_blend(y, preds_dict, cand, n_tries=4000, n_refine=120, seed=123):
    rng = np.random.default_rng(seed)
    P = np.stack([clip01(preds_dict[m]) for m in cand], axis=0)
    m = len(cand)

    best_w = None
    best_mse = None

    for _ in range(n_tries):
        w = rng.dirichlet(np.ones(m))
        p = w @ P
        mse = mean_squared_error(y, p)
        if best_mse is None or mse < best_mse:
            best_mse = mse
            best_w = w

    w = best_w.copy()
    for _ in range(n_refine):
        i = int(rng.integers(0, m))
        j = int(rng.integers(0, m))
        if i == j:
            continue
        delta = float(rng.normal(0.0, 0.05))
        w2 = w.copy()
        w2[i] = max(0.0, w2[i] + delta)
        w2[j] = max(0.0, w2[j] - delta)
        s = w2.sum()
        if s <= 0:
            continue
        w2 /= s
        p2 = w2 @ P
        mse2 = mean_squared_error(y, p2)
        if mse2 < best_mse:
            best_mse = mse2
            w = w2

    return w, float(best_mse)

# ----------------------------
# MAIN
# ----------------------------
def main():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    y_raw = train[TARGETS].to_numpy(dtype=np.float64)
    y_s = clip01(scale_targets_np(y_raw)).astype(np.float32)

    pid_train = train["participant_id"] if "participant_id" in train.columns else None
    pid_test  = test["participant_id"]  if "participant_id" in test.columns else None
    groups = None
    if CV_MODE == "group" and pid_train is not None:
        groups = pid_train.astype(str).to_numpy()

    # Build features
    if CACHE_FEATURES and os.path.exists(CACHE_TRAIN) and os.path.exists(CACHE_TEST):
        Xtr_all, tr_ids, raw_tr, pidtr_cached = pd.read_pickle(CACHE_TRAIN)
        Xte_all, te_ids, raw_te, pidte_cached = pd.read_pickle(CACHE_TEST)
        pid_train_used = pidtr_cached
        pid_test_used  = pidte_cached
        print("Loaded cached features:", CACHE_TAG)
    else:
        Xtr_all, tr_ids, raw_tr, pid_train_used = featurize(train.drop(columns=TARGETS, errors="ignore"), is_train=True)
        Xte_all, te_ids, raw_te, pid_test_used  = featurize(test, is_train=False)

        if CACHE_FEATURES:
            pd.to_pickle((Xtr_all, tr_ids, raw_tr, pid_train_used), CACHE_TRAIN)
            pd.to_pickle((Xte_all, te_ids, raw_te, pid_test_used), CACHE_TEST)
            print("Saved cached features:", CACHE_TAG)

    Xtr = ensure_finite(Xtr_all.to_numpy(dtype=np.float32))
    Xte = ensure_finite(Xte_all.to_numpy(dtype=np.float32))

    blocks_tr = {"tab": Xtr}
    blocks_te = {"tab": Xte}

    if ADD_RAW_FLAT and (raw_tr is not None) and (raw_te is not None):
        raw_tr = ensure_finite(raw_tr.astype(np.float32))
        raw_te = ensure_finite(raw_te.astype(np.float32))

        blocks_tr["raw_flat"] = raw_tr
        blocks_te["raw_flat"] = raw_te

        for d in PCA_DIMS:
            pca = PCA(n_components=d, svd_solver="randomized", random_state=RANDOM_STATE)
            trp = pca.fit_transform(raw_tr).astype(np.float32)
            tep = pca.transform(raw_te).astype(np.float32)
            blocks_tr[f"pca_{d}"] = ensure_finite(trp)
            blocks_te[f"pca_{d}"] = ensure_finite(tep)

    X_train = np.concatenate([blocks_tr[k] for k in sorted(blocks_tr.keys())], axis=1)
    X_test  = np.concatenate([blocks_te[k] for k in sorted(blocks_te.keys())], axis=1)
    X_train = ensure_finite(X_train).astype(np.float32)
    X_test  = ensure_finite(X_test).astype(np.float32)

    print("Feature matrix:", "train=", X_train.shape, "test=", X_test.shape)

    # Build OOF models
    print("\n>>> Building OOF preds (ET + RF bagged) ...")
    oof_dict, split_label = build_oof_models(
        X_train, y_s, groups=groups, bag_seeds=OOF_BAG_SEEDS, n_estimators=OOF_N_ESTIMATORS, verbose=True
    )
    print("CV splitter:", split_label)

    if pid_train_used is not None:
        base_g_oof, base_p_oof = oof_baselines(y_s, pid_train_used, groups=groups)
        oof_dict["BASE_GLOBAL"] = base_g_oof
        oof_dict["BASE_PERPID"] = base_p_oof

    # Stacking
    idx = {"angle":0, "depth":1, "left_right":2}
    test_dict = {}

    if USE_STACKING:
        print("\n>>> Ridge stacking per target ...")
        model_names = list(oof_dict.keys())
        meta_models = {}
        oof_meta = np.zeros_like(y_s, dtype=np.float64)

        for t, j in idx.items():
            P = np.stack([clip01(oof_dict[m][:, j]) for m in model_names], axis=1)
            alpha, oof_p, rr = crossfit_ridge(P, y_s[:, j], RIDGE_ALPHAS)
            oof_p = clip01(oof_p)
            oof_meta[:, j] = oof_p
            meta_models[t] = rr
            print(f"[STACK {t}] alpha={alpha} mse={mean_squared_error(y_s[:,j], oof_p):.6f}")

        oof_dict["META"] = oof_meta

    # FINAL models
    print("\n>>> FINAL refit (bigger models on full train) ...")
    
    # ExtraTrees
    for mname, mparams in ET_PARAMS_LIST:
        preds = []
        for i, s in enumerate(FINAL_BAG_SEEDS, start=1):
            model = fit_et(s, FINAL_N_ESTIMATORS, mparams)
            model.fit(X_train, y_s)
            preds.append(model.predict(X_test))
            if i % 3 == 0:
                print(f"  {mname}: seed {i}/{len(FINAL_BAG_SEEDS)} done")
        test_dict[mname] = clip01(np.mean(preds, axis=0)).astype(np.float64)
    
    # RandomForest
    for mname, mparams in RF_PARAMS_LIST:
        preds = []
        for i, s in enumerate(FINAL_BAG_SEEDS, start=1):
            model = fit_rf(s, FINAL_N_ESTIMATORS, mparams)
            model.fit(X_train, y_s)
            preds.append(model.predict(X_test))
            if i % 3 == 0:
                print(f"  {mname}: seed {i}/{len(FINAL_BAG_SEEDS)} done")
        test_dict[mname] = clip01(np.mean(preds, axis=0)).astype(np.float64)

    if pid_train_used is not None and pid_test_used is not None:
        bg, bp = final_baselines(y_s, pid_train_used, pid_test_used)
        test_dict["BASE_GLOBAL"] = clip01(bg).astype(np.float64)
        test_dict["BASE_PERPID"] = clip01(bp).astype(np.float64)

    if USE_STACKING:
        model_names = list(oof_dict.keys())
        model_names.remove("META")
        meta_test = np.zeros((X_test.shape[0], 3), dtype=np.float64)
        for t, j in idx.items():
            Pte = np.stack([clip01(test_dict[m][:, j]) for m in model_names], axis=1)
            meta_test[:, j] = clip01(meta_models[t].predict(Pte))
        test_dict["META"] = meta_test

    # TOPK blend
    if USE_TOPK_BLEND:
        print("\n>>> TOPK-per-target blend optimization ...")
        all_models = list(oof_dict.keys())

        oof_final = np.zeros_like(y_s, dtype=np.float64)
        te_final  = np.zeros((X_test.shape[0], 3), dtype=np.float64)

        for t, j in idx.items():
            scores = []
            for m in all_models:
                scores.append((m, mean_squared_error(y_s[:, j], clip01(oof_dict[m][:, j]))))
            scores.sort(key=lambda x: x[1])
            ranked = [m for m,_ in scores]
            cand = ranked[:TOPK_PER_TARGET]

            if CORR_PRUNE and len(cand) > 2:
                P = np.stack([clip01(oof_dict[m][:, j]) for m in cand], axis=0)
                cand = corr_prune(cand, P, max_corr=MAX_CORR)

            print(f"[{t}] candidates:", cand)

            preds_o = {m: oof_dict[m][:, j] for m in cand}
            w, mse = optimize_blend(
                y_s[:, j], preds_o, cand,
                n_tries=BLEND_RANDOM_TRIES, n_refine=BLEND_REFINES, seed=1000 + j
            )
            P_o = np.stack([clip01(oof_dict[m][:, j]) for m in cand], axis=0)
            P_t = np.stack([clip01(test_dict[m][:, j]) for m in cand], axis=0)

            oof_final[:, j] = w @ P_o
            te_final[:, j]  = w @ P_t

            ww = {cand[k]: float(w[k]) for k in range(len(cand))}
            ww = dict(sorted(ww.items(), key=lambda x: -x[1]))
            print(f"[{t}] oof_mse={mse:.6f} weights:", {k: round(v,4) for k,v in ww.items() if v > 1e-3})

        pred_test = clip01(te_final)
    else:
        if "META" in test_dict:
            pred_test = clip01(test_dict["META"])
        else:
            pred_test = clip01(np.mean([test_dict[k] for k in test_dict.keys()], axis=0))

    # Diagnostics
    df_pred = pd.DataFrame(pred_test, columns=["scaled_angle","scaled_depth","scaled_left_right"])
    print("\nTest prediction diagnostics:")
    print("  nunique:", df_pred.nunique().to_dict())
    print("  std:", df_pred.std().to_dict())
    print("  min/max:\n", pd.concat([df_pred.min(), df_pred.max()], axis=1, keys=["min","max"]))

    # Write submission
    submission = pd.DataFrame({
        "id": te_ids.to_numpy() if isinstance(te_ids, pd.Series) else np.asarray(te_ids),
        "scaled_angle": pred_test[:, 0],
        "scaled_depth": pred_test[:, 1],
        "scaled_left_right": pred_test[:, 2],
    })
    submission.to_csv(OUT_PATH, index=False)
    print("\nWrote:", OUT_PATH)
    print(submission.head())

if __name__ == "__main__":
    main()
