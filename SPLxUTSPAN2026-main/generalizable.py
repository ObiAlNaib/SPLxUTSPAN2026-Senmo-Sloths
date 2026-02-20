#!/usr/bin/env python3
"""
shot_predictor_v3.py

Implements ALL 7 improvement suggestions + detailed diagnostics.

Summary of what's included:

1) Put height/scale back explicitly:
   - Keep per-frame features normalized by torso scale s (neck-midhip length),
   - BUT also feed global scalars to the model: [s, release_x, release_y, release_z, release_speed, trunk_lean_deg].

2) Improve release-time model:
   - Larger candidate window (±15 frames) around the provided release frame,
   - Release weights computed by attention over candidate hidden states,
   - Sharpness regularizer (entropy penalty) to encourage confident selection.

3) Domain generalization (participant invariance):
   - Gradient Reversal Layer (GRL) + participant classifier head.
   - Trains representation to be less participant-identifiable.

4) Mirror augmentation across y-axis:
   - Reflect y -> -y
   - Swap left/right body blocks (arms, legs, hand orientation),
   - Flip left_right label sign,
   - Flip phi sign in analytic target,
   - Swap "release hand" from right->left by swapping candidate release points and analytic targets.

5) Add trunk + legs cues:
   - Adds hips/knees/ankles rel(midhip) + velocities,
   - Adds trunk lean (deg) and trunk lean angular velocity (deg/s).

6) Reweight losses with angle emphasis:
   - Rim loss weights angle more strongly,
   - Stabilized Stage 2 uses: required-param loss (strong) + small rim loss + vtgt anchor + offset reg + release entropy reg.

7) Non-ensemble robustness:
   - Optional MC-dropout inference + test-time time-jitter averaging.
   - Enable with --mc_dropout N and --tta_jitter J.

Outputs:
- predictions CSV with per-shot metrics, params, offsets, release entropy, and diagnostic deltas.
- Console fold summaries with:
  - mean±std absolute errors (depth/lr/angle),
  - correlations,
  - parameter MAE/bias vs vtgt and vs required-params-from-predicted-release,
  - top-K worst shots per fold (configurable).

Run:
    python shot_predictor_v3.py --train_csv train.csv --release_csv train_release_frames.csv --out_csv predictions_lopo.csv

Optional diagnostics knobs:
    --topk_worst 10
    --verbose 1
    --mc_dropout 20 --tta_jitter 2
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Constants
# -------------------------

G_FTPS2 = 32.174
FPS = 60.0
Z_RIM_FT = 10.0

# Dataset definition anchors (feet)
X0_DEPTH_FT = 6.0      # depth=0 at x=6
Y0_LR_FT = -25.0       # left_right=0 at y=-25

# Context window (increased)
PRE_FRAMES = 90
POST_FRAMES = 15

# Release candidate window (improved)
RELEASE_K = 15
RELEASE_M = 2 * RELEASE_K + 1

# Smoothing
POS_SMOOTH = 5

# Training
SEED = 7
BATCH_SIZE = 32
LR = 8e-4
WEIGHT_DECAY = 2e-5

PRETRAIN_EPOCHS_MAX = 160
FINETUNE_EPOCHS_MAX = 320
PATIENCE = 28

# Transformer
D_MODEL = 192
N_HEAD = 8
N_LAYERS = 4
FF_MULT = 4
DROPOUT = 0.20

# Parameter ranges
V0_MIN = 10.0
V0_MAX = 45.0
THETA_MIN_DEG = 25.0
THETA_MAX_DEG = 80.0
PHI_MAX_DEG = 25.0

# Augmentations
TIME_JITTER = 3          # train-time window jitter (frames)
NOISE_STD = 0.02         # train-time Gaussian noise on normalized features
MIRROR_PROB = 0.50       # mirror augmentation probability

# Loss weights (Stage 2)
W_REQPARAM = 1.00        # strong: predicted params vs required params (from predicted r0 + labels)
W_RIM = 0.35             # small: rim metrics
W_VTGT = 0.25            # weak anchor to vtgt (analytic at nominal release centroid)
W_OFFSET_REG = 0.15      # penalize large learned ball offsets
W_RELEASE_ENT = 0.02     # sharpen release weights

# Angle emphasis in rim loss
ANGLE_WEIGHT = 2.5
DEPTH_WEIGHT = 1.0
LR_WEIGHT = 1.0

# Feasibility penalties
INVALID_PENALTY = 60.0

# Domain adversarial (participant invariance)
ADV_W_PRETRAIN = 0.10
ADV_W_FINETUNE = 0.15
GRL_LAMBDA = 1.0

# Detailed outputs
DEFAULT_TOPK_WORST = 8


# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_parse_list(cell) -> Optional[List[float]]:
    if cell is None:
        return None
    if isinstance(cell, (list, tuple, np.ndarray)):
        return list(cell)

    s = str(cell).strip()
    if not s:
        return None

    if s.startswith("array(") and s.endswith(")"):
        s = s[6:-1].strip()

    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return list(v)
    except Exception:
        pass

    s2 = s.replace("NaN", "null").replace("nan", "null")
    try:
        v = json.loads(s2)
        if isinstance(v, list):
            return v
    except Exception:
        return None

    return None


def to_float_array(x: Optional[List[float]], target_len: int) -> np.ndarray:
    if x is None:
        return np.full((target_len,), np.nan, dtype=np.float32)

    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size < target_len:
        pad_val = arr[-1] if arr.size > 0 else np.nan
        pad = np.full((target_len - arr.size,), pad_val, dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.size > target_len:
        arr = arr[:target_len]

    arr[~np.isfinite(arr)] = np.nan
    return arr


def fill_nans_1d(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=True)
    idx = np.arange(a.size)
    mask = np.isfinite(a)
    if not np.any(mask):
        return np.zeros_like(a)
    a[~mask] = np.interp(idx[~mask], idx[mask], a[mask]).astype(np.float32)
    return a


def smooth_ma(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    if x.ndim == 1:
        x2 = x[:, None]
    else:
        x2 = x

    T, D = x2.shape
    pad = k // 2
    xpad = np.pad(x2, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones((k,), dtype=np.float32) / float(k)

    y = np.empty((T, D), dtype=np.float32)
    for d in range(D):
        y[:, d] = np.convolve(xpad[:, d], kernel, mode="valid").astype(np.float32)

    if x.ndim == 1:
        return y[:, 0]
    return y


def idx_clamp(i: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, i))


def window_indices(center: int, pre: int, post: int, lo: int, hi: int) -> List[int]:
    return [idx_clamp(t, lo, hi) for t in range(center - pre, center + post + 1)]


def unit(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def safe_angle_between(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-6 or nv < 1e-6:
        return 0.0
    c = float(np.dot(u, v) / (nu * nv))
    c = max(-1.0, min(1.0, c))
    return float(math.acos(c))


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return 0.0
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def shift_time(x: torch.Tensor, offset: int) -> torch.Tensor:
    """
    x: [B, T, F]
    offset > 0 means shift earlier frames later (pad front with first frame).
    """
    if offset == 0:
        return x
    B, T, F = x.shape
    if offset > 0:
        pad = x[:, :1, :].expand(B, offset, F)
        return torch.cat([pad, x[:, :T - offset, :]], dim=1)
    off = -offset
    pad = x[:, -1:, :].expand(B, off, F)
    return torch.cat([x[:, off:, :], pad], dim=1)


# -------------------------
# Data structure
# -------------------------

@dataclass
class Example:
    shot_id: str
    participant_id: int
    pid_index: int

    x_seq: np.ndarray                 # [T, F] (already normalized by torso scale for positional/vel features)
    g_vec: np.ndarray                 # [G] global scalars (un-normalized; normalized later)

    # Release candidates + indices into the window for attention gather
    cand_pos_in_window: np.ndarray    # [M] indices in 0..T-1
    r_cand_right: np.ndarray          # [M, 3] feet (right hand centroid candidates)
    r_cand_left: np.ndarray           # [M, 3] feet (left hand centroid candidates)

    # For offset scaling
    scale_s: float                    # torso scale (feet)

    # Labels
    y_depth_in: float
    y_lr_in: float
    y_angle_deg: float

    # Analytic velocity targets (v0, theta, phi) computed at nominal release centroid:
    vtgt_right: np.ndarray            # [3]
    vtgt_left: np.ndarray             # [3]
    vtgt_right_valid: bool
    vtgt_left_valid: bool


# -------------------------
# Feature builder
# -------------------------

class FeatureBuilder:
    """
    Produces:
    - per-frame feature vectors x_seq (pos/vel normalized by torso scale s),
    - global scalars g_vec (keeps absolute height/position info),
    - right and left hand release candidates,
    - analytic targets for both hands.
    """

    def __init__(self, df: pd.DataFrame, frames_total: int = 240):
        self.df = df
        self.frames_total = frames_total

        # Joints (includes legs/trunk)
        self.joints = [
            "mid_hip", "neck",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle",
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
        ]
        self.joint_cols = self._find_joint_cols(df.columns, self.joints)

        # Hand points
        self.right_hand_bases = self._find_hand_point_bases(df.columns, side="right")
        self.left_hand_bases = self._find_hand_point_bases(df.columns, side="left")

        if len(self.right_hand_bases) == 0:
            raise RuntimeError("No right-hand bases found (right + finger/pinky/thumb with _x/_y/_z).")

        self.right_thumb_idx = [i for i, b in enumerate(self.right_hand_bases) if "thumb" in b.lower()]
        self.right_pinky_idx = [i for i, b in enumerate(self.right_hand_bases) if "pinky" in b.lower()]

        self.left_thumb_idx = [i for i, b in enumerate(self.left_hand_bases) if "thumb" in b.lower()]
        self.left_pinky_idx = [i for i, b in enumerate(self.left_hand_bases) if "pinky" in b.lower()]

        # Feature layout (per-frame)
        # See build_example() order; we store indices to support mirroring.
        self.feature_dim = self._compute_feature_dim()

        # Global scalar feature dimension
        # [s, release_x, release_y, release_z, release_speed, trunk_lean_deg]
        self.g_dim = 6

        # Precompute slices for swapping blocks during mirror augmentation
        self.layout = FeatureLayout()

    @staticmethod
    def _find_joint_cols(columns: Sequence[str], joints: Sequence[str]) -> Dict[str, Dict[str, str]]:
        colset = set(columns)
        out: Dict[str, Dict[str, str]] = {}
        for j in joints:
            out[j] = {}
            for axis in ("x", "y", "z"):
                col = f"{j}_{axis}"
                if col in colset:
                    out[j][axis] = col
        return out

    @staticmethod
    def _find_hand_point_bases(columns: Sequence[str], side: str) -> List[str]:
        colset = set(columns)
        bases = set()
        for c in columns:
            cl = c.lower()
            if not (cl.endswith("_x") or cl.endswith("_y") or cl.endswith("_z")):
                continue
            if side not in cl:
                continue
            if ("finger" not in cl) and ("pinky" not in cl) and ("thumb" not in cl):
                continue
            base = c[:-2]
            if (base + "_x") in colset and (base + "_y") in colset and (base + "_z") in colset:
                bases.add(base)
        return sorted(bases)

    def _compute_feature_dim(self) -> int:
        # Matches FeatureLayout definition below
        return FeatureLayout.FEATURE_DIM

    def _parse_xyz(self, row: pd.Series, name: str) -> np.ndarray:
        cols = self.joint_cols.get(name, {})
        if not all(a in cols for a in ("x", "y", "z")):
            return np.zeros((self.frames_total, 3), dtype=np.float32)

        fx = fill_nans_1d(to_float_array(safe_parse_list(row[cols["x"]]), self.frames_total))
        fy = fill_nans_1d(to_float_array(safe_parse_list(row[cols["y"]]), self.frames_total))
        fz = fill_nans_1d(to_float_array(safe_parse_list(row[cols["z"]]), self.frames_total))
        return np.stack([fx, fy, fz], axis=1).astype(np.float32)

    def _parse_hand_points(self, row: pd.Series, bases: List[str]) -> np.ndarray:
        P = len(bases)
        T = self.frames_total
        X = np.zeros((T, P), dtype=np.float32)
        Y = np.zeros((T, P), dtype=np.float32)
        Z = np.zeros((T, P), dtype=np.float32)

        for j, base in enumerate(bases):
            fx = fill_nans_1d(to_float_array(safe_parse_list(row[base + "_x"]), T))
            fy = fill_nans_1d(to_float_array(safe_parse_list(row[base + "_y"]), T))
            fz = fill_nans_1d(to_float_array(safe_parse_list(row[base + "_z"]), T))
            X[:, j] = fx
            Y[:, j] = fy
            Z[:, j] = fz

        return np.stack([X, Y, Z], axis=2).astype(np.float32)  # [T, P, 3]

    @staticmethod
    def _centroid(points: np.ndarray) -> np.ndarray:
        return np.mean(points, axis=1).astype(np.float32)  # [T, 3]

    @staticmethod
    def _pca_normal(points_t: np.ndarray) -> np.ndarray:
        c = points_t - np.mean(points_t, axis=0, keepdims=True)
        cov = c.T @ c
        w, v = np.linalg.eigh(cov.astype(np.float64))
        n = v[:, 0].astype(np.float32)
        return unit(n)

    @staticmethod
    def _hand_orientation_series(points: np.ndarray, thumb_idx: List[int], pinky_idx: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        T, P, _ = points.shape
        normal_u = np.zeros((T, 3), dtype=np.float32)
        axis_u = np.zeros((T, 3), dtype=np.float32)

        prev_n = None
        prev_a = None

        for t in range(T):
            pts = points[t]
            n = FeatureBuilder._pca_normal(pts)
            if prev_n is not None and float(np.dot(n, prev_n)) < 0.0:
                n = (-n).astype(np.float32)
            prev_n = n
            normal_u[t] = n

            if thumb_idx and pinky_idx:
                thumb_c = np.mean(pts[thumb_idx], axis=0)
                pinky_c = np.mean(pts[pinky_idx], axis=0)
                a = unit(thumb_c - pinky_c)
            else:
                a = np.zeros((3,), dtype=np.float32)

            if prev_a is not None and float(np.dot(a, prev_a)) < 0.0:
                a = (-a).astype(np.float32)
            prev_a = a
            axis_u[t] = a

        normal_u = smooth_ma(normal_u, POS_SMOOTH)
        axis_u = smooth_ma(axis_u, POS_SMOOTH)
        for t in range(T):
            normal_u[t] = unit(normal_u[t])
            axis_u[t] = unit(axis_u[t])

        return normal_u, axis_u

    @staticmethod
    def labels_to_rim_xy(depth_in: float, lr_in: float) -> Tuple[float, float]:
        x_rim = X0_DEPTH_FT - (depth_in / 12.0)
        y_rim = Y0_LR_FT + (lr_in / 12.0)
        return float(x_rim), float(y_rim)

    @staticmethod
    def analytic_velocity_target(r0: np.ndarray, depth_in: float, lr_in: float, angle_deg: float) -> Tuple[np.ndarray, bool]:
        x_rim, y_rim = FeatureBuilder.labels_to_rim_xy(depth_in, lr_in)
        dx = x_rim - float(r0[0])
        dy = y_rim - float(r0[1])
        dz = Z_RIM_FT - float(r0[2])

        d_h = math.sqrt(dx * dx + dy * dy)
        a = math.radians(float(angle_deg))
        tan_a = math.tan(a)

        S = dz + d_h * tan_a
        if S <= 1e-6:
            return np.zeros((3,), dtype=np.float32), False

        t2 = (2.0 * S) / G_FTPS2
        if t2 <= 1e-6:
            return np.zeros((3,), dtype=np.float32), False

        t = math.sqrt(t2)
        if t <= 1e-6:
            return np.zeros((3,), dtype=np.float32), False

        vx = dx / t
        vy = dy / t
        v_h = d_h / t
        vz0 = (G_FTPS2 * t) - (v_h * tan_a)

        v0 = math.sqrt(vx * vx + vy * vy + vz0 * vz0)
        theta = math.degrees(math.atan2(vz0, max(1e-6, v_h)))
        phi = math.degrees(math.atan2(vy, -vx))

        if not (math.isfinite(v0) and math.isfinite(theta) and math.isfinite(phi)):
            return np.zeros((3,), dtype=np.float32), False

        return np.array([v0, theta, phi], dtype=np.float32), True

    def build_example(self, row: pd.Series, release_frame: int, pid_index: int) -> Example:
        shot_id = str(row["id"])
        participant_id = int(row["participant_id"])

        y_depth_in = float(row["depth"])
        y_lr_in = float(row["left_right"])
        y_angle_deg = float(row["angle"])

        # Parse joints
        mid = smooth_ma(self._parse_xyz(row, "mid_hip"), POS_SMOOTH)
        neck = smooth_ma(self._parse_xyz(row, "neck"), POS_SMOOTH)

        lhip = smooth_ma(self._parse_xyz(row, "left_hip"), POS_SMOOTH)
        rhip = smooth_ma(self._parse_xyz(row, "right_hip"), POS_SMOOTH)
        lknee = smooth_ma(self._parse_xyz(row, "left_knee"), POS_SMOOTH)
        rknee = smooth_ma(self._parse_xyz(row, "right_knee"), POS_SMOOTH)
        lank = smooth_ma(self._parse_xyz(row, "left_ankle"), POS_SMOOTH)
        rank = smooth_ma(self._parse_xyz(row, "right_ankle"), POS_SMOOTH)

        lsho = smooth_ma(self._parse_xyz(row, "left_shoulder"), POS_SMOOTH)
        rsho = smooth_ma(self._parse_xyz(row, "right_shoulder"), POS_SMOOTH)
        lel = smooth_ma(self._parse_xyz(row, "left_elbow"), POS_SMOOTH)
        rel = smooth_ma(self._parse_xyz(row, "right_elbow"), POS_SMOOTH)
        lwr = smooth_ma(self._parse_xyz(row, "left_wrist"), POS_SMOOTH)
        rwr = smooth_ma(self._parse_xyz(row, "right_wrist"), POS_SMOOTH)

        # Hands
        rh_pts = self._parse_hand_points(row, self.right_hand_bases)
        rh_pts = smooth_ma(rh_pts.reshape(self.frames_total, -1), POS_SMOOTH).reshape(rh_pts.shape)
        rh_cent = self._centroid(rh_pts)

        if len(self.left_hand_bases) > 0:
            lh_pts = self._parse_hand_points(row, self.left_hand_bases)
            lh_pts = smooth_ma(lh_pts.reshape(self.frames_total, -1), POS_SMOOTH).reshape(lh_pts.shape)
            lh_cent = self._centroid(lh_pts)
        else:
            lh_pts = np.zeros((self.frames_total, 1, 3), dtype=np.float32)
            lh_cent = np.zeros((self.frames_total, 3), dtype=np.float32)

        # Hand orientation
        rh_n, rh_a = self._hand_orientation_series(rh_pts, self.right_thumb_idx, self.right_pinky_idx)
        if len(self.left_hand_bases) > 0:
            lh_n, lh_a = self._hand_orientation_series(lh_pts, self.left_thumb_idx, self.left_pinky_idx)
        else:
            lh_n = np.zeros((self.frames_total, 3), dtype=np.float32)
            lh_a = np.zeros((self.frames_total, 3), dtype=np.float32)

        # Velocities
        def vel(x: np.ndarray) -> np.ndarray:
            return np.gradient(x, axis=0) * FPS

        rh_v = vel(rh_cent)
        lh_v = vel(lh_cent)

        rwr_v = vel(rwr)
        rel_v = vel(rel)
        rsho_v = vel(rsho)

        lwr_v = vel(lwr)
        lel_v = vel(lel)
        lsho_v = vel(lsho)

        mid_v = vel(mid)
        neck_v = vel(neck)

        lhip_v = vel(lhip)
        rhip_v = vel(rhip)
        lknee_v = vel(lknee)
        rknee_v = vel(rknee)
        lank_v = vel(lank)
        rank_v = vel(rank)

        rh_dn = vel(rh_n)
        rh_da = vel(rh_a)
        lh_dn = vel(lh_n)
        lh_da = vel(lh_a)

        # Joint angles (right and left)
        wrist_ang_r = np.zeros((self.frames_total,), dtype=np.float32)
        elbow_ang_r = np.zeros((self.frames_total,), dtype=np.float32)
        wrist_ang_l = np.zeros((self.frames_total,), dtype=np.float32)
        elbow_ang_l = np.zeros((self.frames_total,), dtype=np.float32)

        for t in range(self.frames_total):
            wrist_ang_r[t] = safe_angle_between(rh_cent[t] - rwr[t], rel[t] - rwr[t])
            elbow_ang_r[t] = safe_angle_between(rwr[t] - rel[t], rsho[t] - rel[t])
            wrist_ang_l[t] = safe_angle_between(lh_cent[t] - lwr[t], lel[t] - lwr[t])
            elbow_ang_l[t] = safe_angle_between(lwr[t] - lel[t], lsho[t] - lel[t])

        wrist_ang_r_v = np.gradient(wrist_ang_r) * FPS
        elbow_ang_r_v = np.gradient(elbow_ang_r) * FPS
        wrist_ang_l_v = np.gradient(wrist_ang_l) * FPS
        elbow_ang_l_v = np.gradient(elbow_ang_l) * FPS

        # Trunk lean (deg): angle between trunk vector and vertical (z-axis)
        trunk_lean = np.zeros((self.frames_total,), dtype=np.float32)
        for t in range(self.frames_total):
            trunk = neck[t] - mid[t]
            # angle to vertical => use projection vs z axis
            trunk_u = unit(trunk)
            c = float(np.dot(trunk_u, np.array([0.0, 0.0, 1.0], dtype=np.float32)))
            c = max(-1.0, min(1.0, c))
            trunk_lean[t] = float(math.degrees(math.acos(c)))
        trunk_lean_v = np.gradient(trunk_lean) * FPS

        # Release frame clamp
        fr = idx_clamp(int(release_frame), 0, self.frames_total - 1)

        # Torso scale
        s = float(np.linalg.norm(neck[fr] - mid[fr]))
        if not math.isfinite(s) or s < 1e-3:
            s = 1.0

        # Window indices
        idxs = window_indices(fr, PRE_FRAMES, POST_FRAMES, 0, self.frames_total - 1)
        T = len(idxs)

        # Candidate indices (absolute) and positions in window
        cand_abs = [idx_clamp(fr + d, 0, self.frames_total - 1) for d in range(-RELEASE_K, RELEASE_K + 1)]
        cand_pos_in_window = np.array([idxs.index(t) for t in cand_abs], dtype=np.int64)  # always found since window includes ±K

        # Release candidates (feet)
        r_cand_right = rh_cent[cand_abs, :].astype(np.float32)
        r_cand_left = lh_cent[cand_abs, :].astype(np.float32)

        # Analytic targets for both hands (nominal release centroid)
        vtgt_r, ok_r = self.analytic_velocity_target(rh_cent[fr], y_depth_in, y_lr_in, y_angle_deg)
        vtgt_l, ok_l = self.analytic_velocity_target(lh_cent[fr], y_depth_in, y_lr_in, y_angle_deg)

        # Global scalars (keep absolute info)
        release_speed = float(np.linalg.norm(rh_v[fr]))
        g_vec = np.array(
            [
                s,
                float(rh_cent[fr, 0]),
                float(rh_cent[fr, 1]),
                float(rh_cent[fr, 2]),
                release_speed,
                float(trunk_lean[fr]),
            ],
            dtype=np.float32
        )

        # Build per-frame feature vectors normalized by s for positional/vel components
        x_seq = np.zeros((T, self.feature_dim), dtype=np.float32)
        for i, t in enumerate(idxs):
            mid_t = mid[t]

            # All positional/velocity vectors are normalized by s
            feat = np.concatenate(
                [
                    (rh_cent[t] / s), (rh_v[t] / s),                 # RH pos/vel 6
                    (lh_cent[t] / s), (lh_v[t] / s),                 # LH pos/vel 6

                    ((rwr[t] - mid_t) / s), (rwr_v[t] / s),          # RW 6
                    ((rel[t] - mid_t) / s), (rel_v[t] / s),          # RE 6
                    ((rsho[t] - mid_t) / s), (rsho_v[t] / s),        # RS 6

                    ((lwr[t] - mid_t) / s), (lwr_v[t] / s),          # LW 6
                    ((lel[t] - mid_t) / s), (lel_v[t] / s),          # LE 6
                    ((lsho[t] - mid_t) / s), (lsho_v[t] / s),        # LS 6

                    (mid_t / s),                                      # MID pos 3
                    ((neck[t] - mid_t) / s), (neck_v[t] / s),        # NECK rel + vel 6

                    ((lhip[t] - mid_t) / s), (lhip_v[t] / s),        # LHIP 6
                    ((rhip[t] - mid_t) / s), (rhip_v[t] / s),        # RHIP 6

                    ((lknee[t] - mid_t) / s), (lknee_v[t] / s),      # LKNEE 6
                    ((rknee[t] - mid_t) / s), (rknee_v[t] / s),      # RKNEE 6

                    ((lank[t] - mid_t) / s), (lank_v[t] / s),        # LANK 6
                    ((rank[t] - mid_t) / s), (rank_v[t] / s),        # RANK 6

                    np.array(                                         # joint angles (rad) + rates
                        [
                            wrist_ang_r[t], wrist_ang_r_v[t],
                            elbow_ang_r[t], elbow_ang_r_v[t],
                            wrist_ang_l[t], wrist_ang_l_v[t],
                            elbow_ang_l[t], elbow_ang_l_v[t],
                        ],
                        dtype=np.float32
                    ),                                                # 8

                    np.array([trunk_lean[t], trunk_lean_v[t]], dtype=np.float32),  # 2

                    rh_n[t], rh_dn[t], rh_a[t], rh_da[t],             # 12
                    lh_n[t], lh_dn[t], lh_a[t], lh_da[t],             # 12
                ],
                axis=0
            ).astype(np.float32)

            x_seq[i] = feat

        return Example(
            shot_id=shot_id,
            participant_id=participant_id,
            pid_index=pid_index,
            x_seq=x_seq,
            g_vec=g_vec,
            cand_pos_in_window=cand_pos_in_window,
            r_cand_right=r_cand_right,
            r_cand_left=r_cand_left,
            scale_s=s,
            y_depth_in=y_depth_in,
            y_lr_in=y_lr_in,
            y_angle_deg=y_angle_deg,
            vtgt_right=vtgt_r,
            vtgt_left=vtgt_l,
            vtgt_right_valid=ok_r,
            vtgt_left_valid=ok_l,
        )


# -------------------------
# Feature layout (for mirroring)
# -------------------------

class FeatureLayout:
    """
    Defines feature ordering and swap blocks for left/right mirroring augmentation.
    """

    # Per-frame feature dimension (must match build_example order)
    FEATURE_DIM = 127

    # Indices for blocks (start, end) in feature vector
    # Each vector block is [pos(3), vel(3)] => 6
    RH = slice(0, 6)
    LH = slice(6, 12)

    RW = slice(12, 18)
    RE = slice(18, 24)
    RS = slice(24, 30)

    LW = slice(30, 36)
    LE = slice(36, 42)
    LS = slice(42, 48)

    MID = slice(48, 51)
    NECK = slice(51, 57)

    LHIP = slice(57, 63)
    RHIP = slice(63, 69)

    LKNEE = slice(69, 75)
    RKNEE = slice(75, 81)

    LANK = slice(81, 87)
    RANK = slice(87, 93)

    # Scalars:
    ANG = slice(93, 101)     # 8
    TRUNK = slice(101, 103)  # 2

    # Hand orientation blocks (12 each)
    RH_ORI = slice(103, 115)
    LH_ORI = slice(115, 127)

    # Indices of all y-components within vector blocks (pos/vel/rel vectors and orientation vectors)
    # We compute these programmatically from the slices above.
    def y_component_indices(self) -> np.ndarray:
        idxs: List[int] = []

        def add_y_for_slice(s: slice, dims: int = 6):
            # pos y is offset 1, vel y is offset 4 in [x,y,z,vx,vy,vz]
            base = s.start
            idxs.append(base + 1)
            idxs.append(base + 4)

        for s in [self.RH, self.LH, self.RW, self.RE, self.RS, self.LW, self.LE, self.LS,
                  self.NECK, self.LHIP, self.RHIP, self.LKNEE, self.RKNEE, self.LANK, self.RANK]:
            add_y_for_slice(s)

        # MID is pos only: [x,y,z]
        idxs.append(self.MID.start + 1)

        # Orientation blocks are 12 = [nx,ny,nz, dnx,dny,dnz, ax,ay,az, dax,day,daz]
        def add_y_for_ori(s: slice):
            base = s.start
            idxs.extend([base + 1, base + 4, base + 7, base + 10])

        add_y_for_ori(self.RH_ORI)
        add_y_for_ori(self.LH_ORI)

        return np.asarray(sorted(set(idxs)), dtype=np.int64)

    # Swap pairs for mirroring left<->right
    def swap_pairs(self) -> List[Tuple[slice, slice]]:
        return [
            (self.RH, self.LH),
            (self.RW, self.LW),
            (self.RE, self.LE),
            (self.RS, self.LS),
            (self.RHIP, self.LHIP),
            (self.RKNEE, self.LKNEE),
            (self.RANK, self.LANK),
            (self.RH_ORI, self.LH_ORI),
        ]

    # Swap scalar angle blocks: right angles <-> left angles inside ANG slice
    # ANG order is [wr,wr_v, er,er_v, wl,wl_v, el,el_v]
    def swap_angle_scalars(self, ang: np.ndarray) -> np.ndarray:
        a = ang.copy()
        # swap first 4 with last 4
        a[0:4], a[4:8] = a[4:8].copy(), a[0:4].copy()
        return a


# -------------------------
# Dataset / Collate (includes mirror augmentation)
# -------------------------

class ShotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        examples: List[Example],
        x_mean: np.ndarray,
        x_std: np.ndarray,
        g_mean: np.ndarray,
        g_std: np.ndarray,
        train_mode: bool,
        layout: FeatureLayout,
    ):
        self.examples = examples
        self.x_mean = x_mean.astype(np.float32)
        self.x_std = x_std.astype(np.float32)
        self.g_mean = g_mean.astype(np.float32)
        self.g_std = g_std.astype(np.float32)
        self.train_mode = train_mode
        self.layout = layout
        self.y_idxs = self.layout.y_component_indices()
        self.swap_pairs = self.layout.swap_pairs()

    def __len__(self) -> int:
        return len(self.examples)

    def _time_jitter(self, x: np.ndarray) -> np.ndarray:
        if TIME_JITTER <= 0:
            return x
        o = random.randint(-TIME_JITTER, TIME_JITTER)
        if o == 0:
            return x
        T = x.shape[0]
        if o > 0:
            pad = np.repeat(x[:1], o, axis=0)
            return np.concatenate([pad, x[:T - o]], axis=0)
        pad = np.repeat(x[-1:], -o, axis=0)
        return np.concatenate([x[-o:], pad], axis=0)

    def _mirror_augment(
        self,
        x: np.ndarray,
        g: np.ndarray,
        y: np.ndarray,
        vtgt: np.ndarray,
        vmask: float,
        r_cand: np.ndarray,
        cand_pos: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        """
        Mirror across y-axis:
        - swap left/right blocks,
        - flip y components,
        - flip left_right label,
        - flip phi in vtgt,
        - flip release candidate y.
        """
        x2 = x.copy()

        # swap left/right blocks
        for a, b in self.swap_pairs:
            tmp = x2[:, a].copy()
            x2[:, a] = x2[:, b]
            x2[:, b] = tmp

        # swap scalar angle entries within ANG slice
        ang = x2[:, self.layout.ANG]
        # vectorized swap
        ang_sw = np.concatenate([ang[:, 4:8], ang[:, 0:4]], axis=1)
        x2[:, self.layout.ANG] = ang_sw

        # flip y-components
        x2[:, self.y_idxs] *= -1.0

        # global scalars g = [s, rel_x, rel_y, rel_z, rel_speed, trunk_lean]
        g2 = g.copy()
        g2[2] *= -1.0  # release_y flips sign

        # labels: y = [depth, left_right, angle]
        y2 = y.copy()
        y2[1] *= -1.0

        # vtgt = [v0, theta, phi] => phi flips
        vt2 = vtgt.copy()
        vt2[2] *= -1.0

        # release candidates y flips
        r2 = r_cand.copy()
        r2[:, 1] *= -1.0

        # cand_pos unchanged
        return x2, g2, y2, vt2, vmask, r2, cand_pos

    def __getitem__(self, idx: int):
        ex = self.examples[idx]

        # Choose "right" as default; mirror augmentation will swap to left-hand candidates by selecting those before mirror
        x = ex.x_seq.copy()
        g = ex.g_vec.copy()
        y = np.array([ex.y_depth_in, ex.y_lr_in, ex.y_angle_deg], dtype=np.float32)

        # Default: use right-hand analytic target and candidates
        vtgt = ex.vtgt_right.astype(np.float32)
        vmask = 1.0 if ex.vtgt_right_valid else 0.0
        r_cand = ex.r_cand_right.astype(np.float32)
        cand_pos = ex.cand_pos_in_window.astype(np.int64)

        # Train-only: time jitter + noise + mirror augmentation (with left-hand swap)
        if self.train_mode:
            x = self._time_jitter(x)

            # Gaussian noise in feature space (normalized features)
            x = x + np.random.normal(0.0, NOISE_STD, size=x.shape).astype(np.float32)

            # Mirror augmentation: switch to left-hand targets/candidates before mirroring
            if random.random() < MIRROR_PROB:
                vtgt = ex.vtgt_left.astype(np.float32)
                vmask = 1.0 if ex.vtgt_left_valid else 0.0
                r_cand = ex.r_cand_left.astype(np.float32)

                x, g, y, vtgt, vmask, r_cand, cand_pos = self._mirror_augment(x, g, y, vtgt, vmask, r_cand, cand_pos)

        # Normalize x and g with training stats
        x = (x - self.x_mean) / self.x_std
        g = (g - self.g_mean) / self.g_std

        return (
            torch.from_numpy(x).float(),                 # [T, F]
            torch.from_numpy(g).float(),                 # [G]
            torch.from_numpy(r_cand).float(),            # [M, 3]
            torch.from_numpy(cand_pos).long(),           # [M]
            torch.tensor([ex.scale_s], dtype=torch.float32),  # [1] feet (for scaling offset)
            torch.from_numpy(y).float(),                 # [3]
            torch.from_numpy(vtgt).float(),              # [3]
            torch.tensor([vmask], dtype=torch.float32),  # [1]
            torch.tensor([ex.pid_index], dtype=torch.long),   # [1]
            ex.shot_id,
            ex.participant_id,
        )


def collate_fn(batch):
    xs, gs, rc, cp, ss, ys, vt, vm, pid_idx, ids, pids = zip(*batch)
    return (
        torch.stack(xs, dim=0),
        torch.stack(gs, dim=0),
        torch.stack(rc, dim=0),
        torch.stack(cp, dim=0),
        torch.stack(ss, dim=0),
        torch.stack(ys, dim=0),
        torch.stack(vt, dim=0),
        torch.stack(vm, dim=0),
        torch.cat(pid_idx, dim=0),
        list(ids),
        list(pids),
    )


# -------------------------
# Physics (torch)
# -------------------------

def release_velocity_from_params(v0: torch.Tensor, theta_rad: torch.Tensor, phi_rad: torch.Tensor) -> torch.Tensor:
    ct = torch.cos(theta_rad)
    st = torch.sin(theta_rad)
    cp = torch.cos(phi_rad)
    sp = torch.sin(phi_rad)

    vx = -v0 * ct * cp
    vy =  v0 * ct * sp
    vz =  v0 * st
    return torch.stack([vx, vy, vz], dim=-1)


def simulate_to_rim(r0: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x0 = r0[:, 0]
    y0 = r0[:, 1]
    z0 = r0[:, 2]

    vx = v[:, 0]
    vy = v[:, 1]
    vz0 = v[:, 2]

    g = torch.tensor(G_FTPS2, device=r0.device, dtype=r0.dtype)

    disc = vz0 * vz0 + 2.0 * g * (z0 - float(Z_RIM_FT))
    disc_clamped = torch.clamp(disc, min=1e-8)
    t_desc = (vz0 + torch.sqrt(disc_clamped)) / g

    x_rim = x0 + vx * t_desc
    y_rim = y0 + vy * t_desc

    return x_rim, y_rim, t_desc, disc


def rim_metrics(r0: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_rim, y_rim, t_desc, disc = simulate_to_rim(r0, v)

    depth_in = (float(X0_DEPTH_FT) - x_rim) * 12.0
    lr_in = (y_rim - float(Y0_LR_FT)) * 12.0

    g = torch.tensor(G_FTPS2, device=r0.device, dtype=r0.dtype)
    vz_t = v[:, 2] - g * t_desc
    v_h = torch.sqrt(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1]) + 1e-6

    angle_rad = torch.atan2(-vz_t, v_h)
    angle_deg = angle_rad * (180.0 / math.pi)

    return depth_in, lr_in, angle_deg, t_desc, disc


def required_params_from_labels_torch(r0: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    depth_in = y[:, 0]
    lr_in = y[:, 1]
    angle_deg = y[:, 2]

    x_rim = float(X0_DEPTH_FT) - (depth_in / 12.0)
    y_rim = float(Y0_LR_FT) + (lr_in / 12.0)

    dx = x_rim - r0[:, 0]
    dy = y_rim - r0[:, 1]
    dz = float(Z_RIM_FT) - r0[:, 2]

    d_h = torch.sqrt(dx * dx + dy * dy) + 1e-8
    a = angle_deg * (math.pi / 180.0)
    tan_a = torch.tan(a)

    S = dz + d_h * tan_a
    valid = (S > 1e-6).float()

    g = torch.tensor(G_FTPS2, device=r0.device, dtype=r0.dtype)
    t2 = (2.0 * torch.clamp(S, min=1e-6)) / g
    t = torch.sqrt(torch.clamp(t2, min=1e-8))

    vx = dx / t
    vy = dy / t

    v_h = d_h / t
    vz0 = (g * t) - (v_h * tan_a)

    v0 = torch.sqrt(vx * vx + vy * vy + vz0 * vz0)
    theta = torch.atan2(vz0, torch.clamp(v_h, min=1e-6)) * (180.0 / math.pi)
    phi = torch.atan2(vy, -vx) * (180.0 / math.pi)

    return v0, theta, phi, valid


# -------------------------
# Domain adversarial (GRL)
# -------------------------

class GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradReverseFn.apply(x, lambd)


# -------------------------
# Model (Transformer + release attention + offset + adversarial head)
# -------------------------

class ShotNet(nn.Module):
    def __init__(self, feature_dim: int, g_dim: int, seq_len: int, num_participants: int):
        super().__init__()
        self.seq_len = seq_len
        self.in_proj = nn.Linear(feature_dim, D_MODEL)

        self.pos = nn.Parameter(torch.zeros(1, seq_len, D_MODEL))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEAD,
            dim_feedforward=FF_MULT * D_MODEL,
            dropout=DROPOUT,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)

        # Pool + global scalars
        self.g_proj = nn.Sequential(
            nn.Linear(g_dim, D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

        self.fuse = nn.Sequential(
            nn.Linear(D_MODEL + D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
        )

        # Velocity head (v0, theta, phi)
        self.head_vel = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, 3),
        )

        # Release attention: query from fused latent, scores via dot-product with candidate hidden states
        self.rel_query = nn.Linear(D_MODEL, D_MODEL)
        self.rel_scale = math.sqrt(float(D_MODEL))

        # Ball-center offset head (normalized), scaled by torso length s (feet)
        self.head_offset = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, 3),
        )

        # Domain adversarial participant classifier
        self.head_pid = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, num_participants),
        )

    def forward(
        self,
        x_seq: torch.Tensor,         # [B,T,F]
        g_vec: torch.Tensor,         # [B,G]
        r_candidates: torch.Tensor,  # [B,M,3]
        cand_pos: torch.Tensor,      # [B,M] positions in window
        scale_s: torch.Tensor,       # [B,1] feet
        grl_lambda: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        # Encode sequence
        h = self.in_proj(x_seq) + self.pos[:, :x_seq.shape[1], :]
        h = self.encoder(h)
        h = self.norm(h)  # [B,T,D]

        # Pool and fuse global scalars
        pooled = torch.mean(h, dim=1)  # [B,D]
        g_emb = self.g_proj(g_vec)     # [B,D]
        z = self.fuse(torch.cat([pooled, g_emb], dim=1))  # [B,D]

        # Velocity params
        raw = self.head_vel(z)
        raw_v0, raw_theta, raw_phi = raw[:, 0], raw[:, 1], raw[:, 2]

        v0 = V0_MIN + torch.sigmoid(raw_v0) * (V0_MAX - V0_MIN)
        theta = THETA_MIN_DEG + torch.sigmoid(raw_theta) * (THETA_MAX_DEG - THETA_MIN_DEG)
        phi = torch.tanh(raw_phi) * PHI_MAX_DEG

        theta_rad = theta * (math.pi / 180.0)
        phi_rad = phi * (math.pi / 180.0)
        v_vec = release_velocity_from_params(v0, theta_rad, phi_rad)

        # Release attention over candidate hidden states
        # Gather candidate states: [B,M,D]
        B, T, D = h.shape
        M = cand_pos.shape[1]
        cand_pos_clamped = torch.clamp(cand_pos, 0, T - 1)
        idx = cand_pos_clamped.unsqueeze(-1).expand(B, M, D)
        h_cand = torch.gather(h, dim=1, index=idx)

        q = self.rel_query(z).unsqueeze(1)  # [B,1,D]
        scores = torch.sum(h_cand * q, dim=-1) / self.rel_scale  # [B,M]
        w_rel = torch.softmax(scores, dim=1)  # [B,M]

        r_hand = torch.sum(w_rel[:, :, None] * r_candidates, dim=1)  # [B,3] feet

        # Offset
        off_norm = torch.tanh(self.head_offset(z))  # [-1,1]
        off_ft = off_norm * scale_s                 # [B,3] feet
        r_ball = r_hand + off_ft

        # Rim metrics
        pred_depth, pred_lr, pred_angle, t_desc, disc = rim_metrics(r_ball, v_vec)

        # Participant classifier (domain adversarial)
        z_adv = grad_reverse(z, grl_lambda) if grl_lambda > 0 else z.detach() * 0.0 + z
        pid_logits = self.head_pid(z_adv)

        return {
            "pred_depth": pred_depth,
            "pred_lr": pred_lr,
            "pred_angle": pred_angle,

            "v0": v0,
            "theta": theta,
            "phi": phi,
            "v_vec": v_vec,

            "r_hand": r_hand,
            "r0": r_ball,
            "off_ft": off_ft,
            "off_norm": off_norm,

            "t_desc": t_desc,
            "disc": disc,

            "w_release": w_rel,
            "pid_logits": pid_logits,
            "z": z,
        }


# -------------------------
# Norm stats
# -------------------------

def compute_norm_stats(examples: List[Example]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.concatenate([ex.x_seq for ex in examples], axis=0)
    x_mean = np.mean(X, axis=0).astype(np.float32)
    x_std = np.std(X, axis=0).astype(np.float32)
    x_std = np.maximum(x_std, 1e-6)

    G = np.stack([ex.g_vec for ex in examples], axis=0)
    g_mean = np.mean(G, axis=0).astype(np.float32)
    g_std = np.std(G, axis=0).astype(np.float32)
    g_std = np.maximum(g_std, 1e-6)

    Y = np.array([[ex.y_depth_in, ex.y_lr_in, ex.y_angle_deg] for ex in examples], dtype=np.float32)
    y_std = np.std(Y, axis=0).astype(np.float32)
    y_std = np.maximum(y_std, 1e-6)

    return x_mean, x_std, g_mean, g_std, y_std


# -------------------------
# Losses
# -------------------------

def velocity_loss(pred: Dict[str, torch.Tensor], vtgt: torch.Tensor, vmask: torch.Tensor) -> torch.Tensor:
    p = torch.stack([pred["v0"], pred["theta"], pred["phi"]], dim=1)
    e = p - vtgt
    per = F.smooth_l1_loss(e, torch.zeros_like(e), reduction="none").mean(dim=1, keepdim=True)
    return (per * vmask).sum() / (vmask.sum() + 1e-6)


def rim_loss(pred: Dict[str, torch.Tensor], y: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
    p = torch.stack([pred["pred_depth"], pred["pred_lr"], pred["pred_angle"]], dim=1)
    e = (p - y) / y_std[None, :]

    # Weighted error emphasizing angle
    w = torch.tensor([DEPTH_WEIGHT, LR_WEIGHT, ANGLE_WEIGHT], device=e.device, dtype=e.dtype)[None, :]
    e = e * w

    core = F.smooth_l1_loss(e, torch.zeros_like(e), reduction="mean")

    disc = pred["disc"]
    t_desc = pred["t_desc"]
    invalid = ((disc <= 0.0) | (t_desc <= 1e-3)).float().mean()

    return core + (INVALID_PENALTY * invalid)


def required_param_loss(pred: Dict[str, torch.Tensor], y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    v0_req, th_req, ph_req, valid = required_params_from_labels_torch(pred["r0"], y)
    p = torch.stack([pred["v0"], pred["theta"], pred["phi"]], dim=1)
    q = torch.stack([v0_req, th_req, ph_req], dim=1)

    e = p - q
    per = F.smooth_l1_loss(e, torch.zeros_like(e), reduction="none").mean(dim=1)
    loss = (per * valid).sum() / (valid.sum() + 1e-6)
    return loss, valid.mean()


def offset_reg_loss(pred: Dict[str, torch.Tensor]) -> torch.Tensor:
    off = pred["off_ft"]  # feet
    return torch.mean(off * off)


def release_entropy_loss(pred: Dict[str, torch.Tensor]) -> torch.Tensor:
    w = pred["w_release"]  # [B,M]
    ent = -torch.sum(w * torch.log(torch.clamp(w, min=1e-8)), dim=1)  # [B]
    return torch.mean(ent)


def adversarial_pid_loss(pred: Dict[str, torch.Tensor], pid_idx: torch.Tensor) -> torch.Tensor:
    logits = pred["pid_logits"]
    return F.cross_entropy(logits, pid_idx)


# -------------------------
# Diagnostics
# -------------------------

def diagnostic_feasibility(examples: List[Example]) -> None:
    ok_r = [ex for ex in examples if ex.vtgt_right_valid]
    ok_l = [ex for ex in examples if ex.vtgt_left_valid]
    if not ok_r:
        print("Diagnostic: 0 feasible shots for right-hand analytic targets (check parsing).")
        return

    V = np.stack([ex.vtgt_right for ex in ok_r], axis=0)

    v0 = V[:, 0]
    th = V[:, 1]
    ph = V[:, 2]

    def q(a: np.ndarray) -> Tuple[float, float, float]:
        return float(np.percentile(a, 5)), float(np.percentile(a, 50)), float(np.percentile(a, 95))

    print(
        "Diagnostic feasibility (analytic solve using labels + nominal release centroid): "
        f"{len(ok_r)}/{len(examples)} feasible ({(len(ok_r) / max(1, len(examples)) * 100.0):.1f}%)."
    )
    print(
        f"Required v0 ft/s (5/50/95%): {q(v0)[0]:.1f}/{q(v0)[1]:.1f}/{q(v0)[2]:.1f} | "
        f"theta deg (5/50/95%): {q(th)[0]:.1f}/{q(th)[1]:.1f}/{q(th)[2]:.1f} | "
        f"phi deg (5/50/95%): {q(ph)[0]:.1f}/{q(ph)[1]:.1f}/{q(ph)[2]:.1f}"
    )
    if len(ok_l) < len(examples):
        print(
            f"Note: left-hand analytic targets feasible for {len(ok_l)}/{len(examples)} "
            f"({(len(ok_l) / max(1, len(examples)) * 100.0):.1f}%). "
            "If this is low, mirror augmentation will be less effective."
        )


# -------------------------
# Train / Eval
# -------------------------

def split_train_val(examples: List[Example], val_frac: float = 0.15) -> Tuple[List[Example], List[Example]]:
    idxs = list(range(len(examples)))
    random.shuffle(idxs)
    n_val = max(1, int(round(len(examples) * val_frac)))
    val_idx = set(idxs[:n_val])
    train = [examples[i] for i in range(len(examples)) if i not in val_idx]
    val = [examples[i] for i in range(len(examples)) if i in val_idx]
    return train, val


@torch.no_grad()
def predict_batch(
    model: nn.Module,
    x: torch.Tensor,
    g: torch.Tensor,
    rc: torch.Tensor,
    cp: torch.Tensor,
    ss: torch.Tensor,
    device: torch.device,
    mc_dropout: int,
    tta_jitter: int,
) -> Dict[str, torch.Tensor]:
    """
    Optional inference-time robustness:
    - MC dropout: run multiple stochastic passes (dropout enabled),
    - TTA jitter: average across time shifts [-tta_jitter..+tta_jitter] (or just {0} if 0).
    """
    shifts = [0] if tta_jitter <= 0 else list(range(-tta_jitter, tta_jitter + 1))
    n_drop = max(1, int(mc_dropout)) if mc_dropout > 0 else 1

    # Enable dropout for MC dropout
    if mc_dropout > 0:
        model.train()
    else:
        model.eval()

    acc: Dict[str, torch.Tensor] = {}
    count = 0

    for _ in range(n_drop):
        for sh in shifts:
            x_in = shift_time(x, sh) if sh != 0 else x
            pred = model(x_in, g, rc, cp, ss, grl_lambda=0.0)

            # Average selected outputs
            keys = ["pred_depth", "pred_lr", "pred_angle", "v0", "theta", "phi", "r0", "r_hand", "off_ft", "w_release", "disc", "t_desc"]
            for k in keys:
                v = pred[k]
                acc[k] = v if k not in acc else (acc[k] + v)
            count += 1

    for k in list(acc.keys()):
        acc[k] = acc[k] / float(count)

    # Keep pid logits unused in inference
    return acc


@torch.no_grad()
def eval_loader(
    model: nn.Module,
    loader,
    device: torch.device,
    mc_dropout: int,
    tta_jitter: int,
    topk_worst: int,
    verbose: int,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []

    all_true = {"depth": [], "lr": [], "angle": []}
    all_pred = {"depth": [], "lr": [], "angle": []}

    abs_depth: List[float] = []
    abs_lr: List[float] = []
    abs_ang: List[float] = []

    # Param monitoring
    v0_abs_vtgt: List[float] = []
    th_abs_vtgt: List[float] = []
    ph_abs_vtgt: List[float] = []
    v0_bias_vtgt: List[float] = []
    th_bias_vtgt: List[float] = []
    ph_bias_vtgt: List[float] = []

    v0_abs_req: List[float] = []
    th_abs_req: List[float] = []
    ph_abs_req: List[float] = []
    v0_bias_req: List[float] = []
    th_bias_req: List[float] = []
    ph_bias_req: List[float] = []

    release_entropies: List[float] = []
    offset_norms_in: List[float] = []

    # For worst-shot listing
    shot_errors: List[Tuple[float, Dict[str, object]]] = []

    for x, g, rc, cp, ss, y, vtgt, vmask, pid_idx, ids, pids in loader:
        x = x.to(device)
        g = g.to(device)
        rc = rc.to(device)
        cp = cp.to(device)
        ss = ss.to(device)
        y = y.to(device)
        vtgt = vtgt.to(device)
        vmask = vmask.to(device)

        pred = predict_batch(model, x, g, rc, cp, ss, device=device, mc_dropout=mc_dropout, tta_jitter=tta_jitter)

        pd = pred["pred_depth"].detach().cpu().numpy()
        pl = pred["pred_lr"].detach().cpu().numpy()
        pa = pred["pred_angle"].detach().cpu().numpy()

        yd = y[:, 0].detach().cpu().numpy()
        yl = y[:, 1].detach().cpu().numpy()
        ya = y[:, 2].detach().cpu().numpy()

        all_true["depth"].extend(list(yd))
        all_true["lr"].extend(list(yl))
        all_true["angle"].extend(list(ya))

        all_pred["depth"].extend(list(pd))
        all_pred["lr"].extend(list(pl))
        all_pred["angle"].extend(list(pa))

        abs_depth_batch = np.abs(pd - yd)
        abs_lr_batch = np.abs(pl - yl)
        abs_ang_batch = np.abs(pa - ya)

        abs_depth.extend(list(abs_depth_batch))
        abs_lr.extend(list(abs_lr_batch))
        abs_ang.extend(list(abs_ang_batch))

        # Release entropy + offset norms
        w = pred["w_release"].detach().cpu().numpy()
        ent = -np.sum(w * np.log(np.clip(w, 1e-8, 1.0)), axis=1)
        release_entropies.extend(list(ent))

        off = pred["off_ft"].detach().cpu().numpy()
        off_norm = np.linalg.norm(off, axis=1) * 12.0  # inches
        offset_norms_in.extend(list(off_norm))

        # vtgt comparisons
        params = np.stack(
            [
                pred["v0"].detach().cpu().numpy(),
                pred["theta"].detach().cpu().numpy(),
                pred["phi"].detach().cpu().numpy(),
            ],
            axis=1
        )
        vt = vtgt.detach().cpu().numpy()
        vm = vmask[:, 0].detach().cpu().numpy() > 0.5
        if np.any(vm):
            diff = params[vm] - vt[vm]
            v0_abs_vtgt.extend(list(np.abs(diff[:, 0])))
            th_abs_vtgt.extend(list(np.abs(diff[:, 1])))
            ph_abs_vtgt.extend(list(np.abs(diff[:, 2])))

            v0_bias_vtgt.extend(list(diff[:, 0]))
            th_bias_vtgt.extend(list(diff[:, 1]))
            ph_bias_vtgt.extend(list(diff[:, 2]))

        # required params comparisons (using predicted r0)
        r0 = pred["r0"]
        v0_req, th_req, ph_req, valid = required_params_from_labels_torch(r0, y)
        valid_np = valid.detach().cpu().numpy() > 0.5
        req = torch.stack([v0_req, th_req, ph_req], dim=1).detach().cpu().numpy()
        if np.any(valid_np):
            diff2 = params[valid_np] - req[valid_np]
            v0_abs_req.extend(list(np.abs(diff2[:, 0])))
            th_abs_req.extend(list(np.abs(diff2[:, 1])))
            ph_abs_req.extend(list(np.abs(diff2[:, 2])))

            v0_bias_req.extend(list(diff2[:, 0]))
            th_bias_req.extend(list(diff2[:, 1]))
            ph_bias_req.extend(list(diff2[:, 2]))

        # rows + worst list
        for i in range(len(ids)):
            combined = float(abs_depth_batch[i] + abs_lr_batch[i] + 2.0 * abs_ang_batch[i])

            row = {
                "id": ids[i],
                "participant_id": int(pids[i]),

                "true_depth": float(yd[i]),
                "pred_depth": float(pd[i]),
                "abs_err_depth": float(abs_depth_batch[i]),

                "true_left_right": float(yl[i]),
                "pred_left_right": float(pl[i]),
                "abs_err_left_right": float(abs_lr_batch[i]),

                "true_angle": float(ya[i]),
                "pred_angle": float(pa[i]),
                "abs_err_angle": float(abs_ang_batch[i]),

                "pred_v0_ftps": float(params[i, 0]),
                "pred_theta_deg": float(params[i, 1]),
                "pred_phi_deg": float(params[i, 2]),

                "vtgt_v0_ftps": float(vt[i, 0]),
                "vtgt_theta_deg": float(vt[i, 1]),
                "vtgt_phi_deg": float(vt[i, 2]),
                "vtgt_valid": bool(vm[i]) if i < vm.shape[0] else True,

                "pred_release_x_ft": float(pred["r0"][i, 0].detach().cpu().item()),
                "pred_release_y_ft": float(pred["r0"][i, 1].detach().cpu().item()),
                "pred_release_z_ft": float(pred["r0"][i, 2].detach().cpu().item()),

                "pred_offset_norm_in": float(off_norm[i]),
                "release_entropy": float(ent[i]),

                "combined_score": combined,
            }
            rows.append(row)
            shot_errors.append((combined, row))

    # Summary helpers
    def mean_std(xs: List[float]) -> Tuple[float, float]:
        if not xs:
            return 0.0, 0.0
        a = np.asarray(xs, dtype=np.float64)
        return float(a.mean()), float(a.std(ddof=0))

    d_m, d_s = mean_std(abs_depth)
    l_m, l_s = mean_std(abs_lr)
    a_m, a_s = mean_std(abs_ang)

    # Correlations
    depth_corr = pearson_corr(np.asarray(all_true["depth"]), np.asarray(all_pred["depth"]))
    lr_corr = pearson_corr(np.asarray(all_true["lr"]), np.asarray(all_pred["lr"]))
    ang_corr = pearson_corr(np.asarray(all_true["angle"]), np.asarray(all_pred["angle"]))

    # Param stats
    v0m_vt, v0s_vt = mean_std(v0_abs_vtgt)
    thm_vt, ths_vt = mean_std(th_abs_vtgt)
    phm_vt, phs_vt = mean_std(ph_abs_vtgt)

    v0b_vt, _ = mean_std(v0_bias_vtgt)
    thb_vt, _ = mean_std(th_bias_vtgt)
    phb_vt, _ = mean_std(ph_bias_vtgt)

    v0m_rq, v0s_rq = mean_std(v0_abs_req)
    thm_rq, ths_rq = mean_std(th_abs_req)
    phm_rq, phs_rq = mean_std(ph_abs_req)

    v0b_rq, _ = mean_std(v0_bias_req)
    thb_rq, _ = mean_std(th_bias_req)
    phb_rq, _ = mean_std(ph_bias_req)

    ent_m, ent_s = mean_std(release_entropies)
    off_m, off_s = mean_std(offset_norms_in)

    # Worst shots
    shot_errors.sort(key=lambda t: t[0], reverse=True)
    worst = shot_errors[:max(0, int(topk_worst))]

    if verbose and worst:
        print(f"  Worst {len(worst)} shots (combined depth+lr+2*angle):")
        for j, (score, r) in enumerate(worst, start=1):
            print(
                f"    {j:02d}. id={r['id']} | score={score:.2f} | "
                f"depth_err={r['abs_err_depth']:.2f} in | lr_err={r['abs_err_left_right']:.2f} in | "
                f"angle_err={r['abs_err_angle']:.2f} deg | "
                f"theta_pred={r['pred_theta_deg']:.2f} vtgt={r['vtgt_theta_deg']:.2f} | "
                f"off={r['pred_offset_norm_in']:.2f} in | ent={r['release_entropy']:.2f}"
            )

    summary = {
        "n": float(len(abs_depth)),
        "depth_mae": d_m,
        "depth_abs_std": d_s,
        "lr_mae": l_m,
        "lr_abs_std": l_s,
        "angle_mae": a_m,
        "angle_abs_std": a_s,

        "depth_corr": depth_corr,
        "lr_corr": lr_corr,
        "angle_corr": ang_corr,

        "v0_mae_vs_vtgt": v0m_vt,
        "theta_mae_vs_vtgt": thm_vt,
        "phi_mae_vs_vtgt": phm_vt,
        "v0_bias_vs_vtgt": v0b_vt,
        "theta_bias_vs_vtgt": thb_vt,
        "phi_bias_vs_vtgt": phb_vt,

        "v0_mae_vs_req": v0m_rq,
        "theta_mae_vs_req": thm_rq,
        "phi_mae_vs_req": phm_rq,
        "v0_bias_vs_req": v0b_rq,
        "theta_bias_vs_req": thb_rq,
        "phi_bias_vs_req": phb_rq,

        "release_entropy_mean": ent_m,
        "release_entropy_std": ent_s,
        "offset_norm_in_mean": off_m,
        "offset_norm_in_std": off_s,
    }

    return summary, rows


def train_fold(
    models_dir: str,
    heldout_pid: int,
    train_examples: List[Example],
    test_examples: List[Example],
    device: torch.device,
    layout: FeatureLayout,
    num_participants: int,
    mc_dropout: int,
    tta_jitter: int,
    topk_worst: int,
    verbose: int,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:

    train_split, val_split = split_train_val(train_examples, val_frac=0.15)
    x_mean, x_std, g_mean, g_std, y_std = compute_norm_stats(train_split)

    ds_train = ShotDataset(train_split, x_mean, x_std, g_mean, g_std, train_mode=True, layout=layout)
    ds_val = ShotDataset(val_split, x_mean, x_std, g_mean, g_std, train_mode=False, layout=layout)
    ds_test = ShotDataset(test_examples, x_mean, x_std, g_mean, g_std, train_mode=False, layout=layout)

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    seq_len = ds_train.examples[0].x_seq.shape[0]
    model = ShotNet(feature_dim=layout.FEATURE_DIM, g_dim=ds_train.examples[0].g_vec.shape[0], seq_len=seq_len, num_participants=num_participants).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    y_std_t = torch.from_numpy(y_std).to(device)

    # -------------------------
    # Stage 1: pretrain to vtgt + adversarial invariance + mild release/offset regularization
    # -------------------------
    best_val = float("inf")
    best_state = None
    bad = 0

    for _epoch in range(1, PRETRAIN_EPOCHS_MAX + 1):
        model.train()
        for x, g, rc, cp, ss, y, vtgt, vmask, pid_idx, _, _ in dl_train:
            x = x.to(device)
            g = g.to(device)
            rc = rc.to(device)
            cp = cp.to(device)
            ss = ss.to(device)
            vtgt = vtgt.to(device)
            vmask = vmask.to(device)
            pid_idx = pid_idx.to(device)

            pred = model(x, g, rc, cp, ss, grl_lambda=GRL_LAMBDA)

            l_v = velocity_loss(pred, vtgt, vmask)
            l_off = offset_reg_loss(pred)
            l_ent = release_entropy_loss(pred)
            l_adv = adversarial_pid_loss(pred, pid_idx)

            loss = l_v + (0.05 * l_off) + (0.02 * l_ent) + (ADV_W_PRETRAIN * l_adv)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.5)
            opt.step()

        # Validation: use stabilized objective (without adversarial)
        model.eval()
        vloss = 0.0
        nb = 0
        with torch.no_grad():
            for x, g, rc, cp, ss, y, vtgt, vmask, pid_idx, _, _ in dl_val:
                x = x.to(device)
                g = g.to(device)
                rc = rc.to(device)
                cp = cp.to(device)
                ss = ss.to(device)
                y = y.to(device)
                vtgt = vtgt.to(device)
                vmask = vmask.to(device)

                pred = model(x, g, rc, cp, ss, grl_lambda=0.0)

                l_req, valid_frac = required_param_loss(pred, y)
                l_rim = rim_loss(pred, y, y_std_t)
                l_vt = velocity_loss(pred, vtgt, vmask)
                l_off = offset_reg_loss(pred)
                l_ent = release_entropy_loss(pred)

                loss = (W_REQPARAM * l_req) + (W_RIM * l_rim) + (W_VTGT * l_vt) + (W_OFFSET_REG * l_off) + (W_RELEASE_ENT * l_ent)
                loss = loss + INVALID_PENALTY * (1.0 - valid_frac)

                vloss += float(loss.detach().cpu().item())
                nb += 1

        vloss = vloss / max(1, nb)
        if vloss + 1e-6 < best_val:
            best_val = vloss
            best_state = {
                "model": model.state_dict(),
                "x_mean": x_mean,
                "x_std": x_std,
                "g_mean": g_mean,
                "g_std": g_std,
                "y_std": y_std,
                "heldout_pid": heldout_pid,
            }
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    # -------------------------
    # Stage 2: stabilized fine-tune with required-param loss + angle-weighted rim + adversarial invariance
    # -------------------------
    best_val2 = float("inf")
    best_state2 = None
    bad = 0

    for _epoch in range(1, FINETUNE_EPOCHS_MAX + 1):
        model.train()
        for x, g, rc, cp, ss, y, vtgt, vmask, pid_idx, _, _ in dl_train:
            x = x.to(device)
            g = g.to(device)
            rc = rc.to(device)
            cp = cp.to(device)
            ss = ss.to(device)
            y = y.to(device)
            vtgt = vtgt.to(device)
            vmask = vmask.to(device)
            pid_idx = pid_idx.to(device)

            pred = model(x, g, rc, cp, ss, grl_lambda=GRL_LAMBDA)

            l_req, valid_frac = required_param_loss(pred, y)
            l_rim = rim_loss(pred, y, y_std_t)
            l_vt = velocity_loss(pred, vtgt, vmask)
            l_off = offset_reg_loss(pred)
            l_ent = release_entropy_loss(pred)
            l_adv = adversarial_pid_loss(pred, pid_idx)

            loss = (W_REQPARAM * l_req) + (W_RIM * l_rim) + (W_VTGT * l_vt) + (W_OFFSET_REG * l_off) + (W_RELEASE_ENT * l_ent) + (ADV_W_FINETUNE * l_adv)
            loss = loss + INVALID_PENALTY * (1.0 - valid_frac)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.5)
            opt.step()

        # Validation
        model.eval()
        vloss = 0.0
        nb = 0
        with torch.no_grad():
            for x, g, rc, cp, ss, y, vtgt, vmask, pid_idx, _, _ in dl_val:
                x = x.to(device)
                g = g.to(device)
                rc = rc.to(device)
                cp = cp.to(device)
                ss = ss.to(device)
                y = y.to(device)
                vtgt = vtgt.to(device)
                vmask = vmask.to(device)

                pred = model(x, g, rc, cp, ss, grl_lambda=0.0)

                l_req, valid_frac = required_param_loss(pred, y)
                l_rim = rim_loss(pred, y, y_std_t)
                l_vt = velocity_loss(pred, vtgt, vmask)
                l_off = offset_reg_loss(pred)
                l_ent = release_entropy_loss(pred)

                loss = (W_REQPARAM * l_req) + (W_RIM * l_rim) + (W_VTGT * l_vt) + (W_OFFSET_REG * l_off) + (W_RELEASE_ENT * l_ent)
                loss = loss + INVALID_PENALTY * (1.0 - valid_frac)

                vloss += float(loss.detach().cpu().item())
                nb += 1

        vloss = vloss / max(1, nb)
        if vloss + 1e-6 < best_val2:
            best_val2 = vloss
            best_state2 = {
                "model": model.state_dict(),
                "x_mean": x_mean,
                "x_std": x_std,
                "g_mean": g_mean,
                "g_std": g_std,
                "y_std": y_std,
                "heldout_pid": heldout_pid,
            }
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state2 is None:
        best_state2 = best_state if best_state is not None else {
            "model": model.state_dict(),
            "x_mean": x_mean,
            "x_std": x_std,
            "g_mean": g_mean,
            "g_std": g_std,
            "y_std": y_std,
            "heldout_pid": heldout_pid,
        }

    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"fold_participant_{heldout_pid}.pt")
    torch.save(best_state2, model_path)

    # Load best and evaluate test with optional MC dropout / TTA jitter
    model.load_state_dict(best_state2["model"])
    test_summary, test_rows = eval_loader(
        model,
        dl_test,
        device=device,
        mc_dropout=mc_dropout,
        tta_jitter=tta_jitter,
        topk_worst=topk_worst,
        verbose=verbose,
    )

    return test_summary, test_rows


# -------------------------
# Main
# -------------------------

def load_release_frames(path: str) -> Dict[str, int]:
    df = pd.read_csv(path)
    if "id" not in df.columns or "frame" not in df.columns:
        raise RuntimeError("train_release_frames.csv must have columns: id, frame")
    return {str(r["id"]): int(r["frame"]) for _, r in df.iterrows()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="train.csv")
    parser.add_argument("--release_csv", type=str, default="train_release_frames.csv")
    parser.add_argument("--out_csv", type=str, default="predictions_lopo.csv")
    parser.add_argument("--models_dir", type=str, default="models")

    parser.add_argument("--mc_dropout", type=int, default=0, help="If >0, enables MC dropout with N passes at eval.")
    parser.add_argument("--tta_jitter", type=int, default=0, help="If >0, averages eval predictions across time shifts [-J..+J].")

    parser.add_argument("--topk_worst", type=int, default=DEFAULT_TOPK_WORST, help="How many worst shots to print per fold (if verbose).")
    parser.add_argument("--verbose", type=int, default=1, help="0/1. If 1, prints worst shots and extra diagnostics.")

    args = parser.parse_args()

    set_seed(SEED)

    df = pd.read_csv(args.train_csv)
    required = {"id", "participant_id", "depth", "left_right", "angle"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"train.csv missing required columns: {missing}")

    release_map = load_release_frames(args.release_csv)

    # Map participant IDs to 0..K-1 for adversarial head
    participant_ids = sorted(df["participant_id"].unique().tolist())
    pid_to_index = {int(pid): i for i, pid in enumerate(participant_ids)}
    num_participants = len(participant_ids)

    builder = FeatureBuilder(df, frames_total=240)
    layout = FeatureLayout()

    examples: List[Example] = []
    for _, row in df.iterrows():
        sid = str(row["id"])
        if sid not in release_map:
            continue
        pid = int(row["participant_id"])
        examples.append(builder.build_example(row, release_map[sid], pid_index=pid_to_index[pid]))

    if not examples:
        raise RuntimeError("No examples built. Check ID match between train.csv and train_release_frames.csv.")

    diagnostic_feasibility(examples)

    participants = sorted({ex.participant_id for ex in examples})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_rows: List[Dict[str, object]] = []

    fold_summaries: List[Tuple[int, Dict[str, float]]] = []

    for heldout in participants:
        train_ex = [ex for ex in examples if ex.participant_id != heldout]
        test_ex = [ex for ex in examples if ex.participant_id == heldout]

        summary, rows = train_fold(
            models_dir=args.models_dir,
            heldout_pid=heldout,
            train_examples=train_ex,
            test_examples=test_ex,
            device=device,
            layout=layout,
            num_participants=num_participants,
            mc_dropout=args.mc_dropout,
            tta_jitter=args.tta_jitter,
            topk_worst=args.topk_worst,
            verbose=args.verbose,
        )

        fold_summaries.append((heldout, summary))
        all_rows.extend(rows)

        n = int(summary["n"])
        print(
            f"heldout={heldout} | n={n} | "
            f"depth_abs_err mean±std = {summary['depth_mae']:.3f}±{summary['depth_abs_std']:.3f} in | "
            f"lr_abs_err mean±std = {summary['lr_mae']:.3f}±{summary['lr_abs_std']:.3f} in | "
            f"angle_abs_err mean±std = {summary['angle_mae']:.3f}±{summary['angle_abs_std']:.3f} deg"
        )
        print(
            f"  corr(depth,lr,angle) = ({summary['depth_corr']:.3f}, {summary['lr_corr']:.3f}, {summary['angle_corr']:.3f})"
        )
        print(
            f"  params vs vtgt: v0 {summary['v0_mae_vs_vtgt']:.3f} ft/s (bias {summary['v0_bias_vs_vtgt']:.3f}), "
            f"theta {summary['theta_mae_vs_vtgt']:.3f}° (bias {summary['theta_bias_vs_vtgt']:.3f}), "
            f"phi {summary['phi_mae_vs_vtgt']:.3f}° (bias {summary['phi_bias_vs_vtgt']:.3f})"
        )
        print(
            f"  params vs req : v0 {summary['v0_mae_vs_req']:.3f} ft/s (bias {summary['v0_bias_vs_req']:.3f}), "
            f"theta {summary['theta_mae_vs_req']:.3f}° (bias {summary['theta_bias_vs_req']:.3f}), "
            f"phi {summary['phi_mae_vs_req']:.3f}° (bias {summary['phi_bias_vs_req']:.3f})"
        )
        print(
            f"  release_entropy mean±std = {summary['release_entropy_mean']:.3f}±{summary['release_entropy_std']:.3f} | "
            f"offset_norm mean±std = {summary['offset_norm_in_mean']:.3f}±{summary['offset_norm_in_std']:.3f} in"
        )

        if heldout == 4:
            print("  NOTE: heldout=4 spotlight. Watch theta bias, offset norms, and release entropy above.")

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(args.out_csv, index=False)

    def overall_abs_err(col_abs: str) -> Tuple[float, float]:
        e = out_df[col_abs].to_numpy(dtype=np.float64)
        return float(e.mean()), float(e.std(ddof=0))

    d_m, d_s = overall_abs_err("abs_err_depth")
    l_m, l_s = overall_abs_err("abs_err_left_right")
    a_m, a_s = overall_abs_err("abs_err_angle")

    print("\nOverall (all held-out folds combined):")
    print(f"depth_abs_err mean±std = {d_m:.3f}±{d_s:.3f} in")
    print(f"lr_abs_err    mean±std = {l_m:.3f}±{l_s:.3f} in")
    print(f"angle_abs_err mean±std = {a_m:.3f}±{a_s:.3f} deg")
    print(f"\nWrote: {args.out_csv}")
    print(f"Models saved under: {args.models_dir}/")

    if args.mc_dropout > 0 or args.tta_jitter > 0:
        print(
            f"Eval robustness enabled: mc_dropout={args.mc_dropout}, tta_jitter={args.tta_jitter} "
            "(this increases eval compute)."
        )


if __name__ == "__main__":
    main()
