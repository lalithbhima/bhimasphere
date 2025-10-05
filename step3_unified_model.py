# step3_unified_model.py
import os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except Exception:
    HAS_SGKF = False

from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import resample

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional libs (gracefully skip if missing)
HAS_XGB = HAS_LGBM = HAS_CAT = True
try:
    from xgboost import XGBClassifier
except Exception:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
except Exception:
    HAS_LGBM = False
try:
    from catboost import CatBoostClassifier
except Exception:
    HAS_CAT = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

import joblib

# --- Optional deep tabular models ---
HAS_TABTRANS = False
try:
    # https://github.com/lucidrains/tab-transformer-pytorch
    from tab_transformer_pytorch import TabTransformer
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TABTRANS = True
except Exception:
    HAS_TABTRANS = False

if HAS_TABTRANS:
    class FocalLoss(torch.nn.Module):
        def __init__(self, gamma=2.0, weight=None):
            super().__init__()
            self.gamma = gamma
            self.weight = weight
        def forward(self, logits, targets):
            ce_loss = torch.nn.functional.cross_entropy(
                logits, targets, weight=self.weight, reduction="none"
            )
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            return focal_loss.mean()

HAS_TABNET = False
try:
    # https://github.com/dreamquark-ai/tabnet
    from pytorch_tabnet.tab_model import TabNetClassifier
    HAS_TABNET = True
except Exception:
    HAS_TABNET = False

class SkDeepTabWrapper:
    """
    Sklearn-like wrapper around TabTransformer for numeric-only.
    Works with tab_transformer_pytorch versions without `return_embeddings`.
    Adapts to whatever the backbone returns.
    """
    def __init__(self, n_features, n_classes, device="cpu",
                 hidden_dim=64, depth=4, heads=8, attn_dropout=0.1, ff_dropout=0.1,
                 lr=1e-3, batch_size=1024, max_epochs=200, patience=20, verbose=False):
        if not HAS_TABTRANS:
            raise RuntimeError("TabTransformer library not available.")
        self.n_features = n_features
        self.n_classes  = n_classes
        self.device = device
        import torch
        from torch import nn
        self.torch = torch
        self.nn = nn

        # backbone without return_embeddings (older versions)
        self.backbone = TabTransformer(
            categories=(),                # no categoricals
            num_continuous=n_features,    # all features numeric
            dim=hidden_dim,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        ).to(device)

        self.head = None  # lazy init after seeing first forward
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.verbose = verbose
        self._best_state = None

    # ----- helpers -----
    def _forward_backbone(self, X, empty_categ):
        out = self.backbone(empty_categ, X)
        # Some versions may return tuple
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    def _ensure_2d_features(self, out):
        # Accept shapes:
        # (B, C) logits directly -> return ("logits", out)
        # (B, D) embeddings -> return ("emb", out)
        # (B, T, D) token embeddings -> mean-pool over T -> ("emb", pooled)
        if out.dim() == 3:
            out = out.mean(dim=1)  # pool tokens -> (B, D)
            return "emb", out
        elif out.dim() == 2:
            # If width equals n_classes, assume logits
            if out.shape[1] == self.n_classes:
                return "logits", out
            return "emb", out
        else:
            raise RuntimeError(f"Unexpected backbone output shape: {tuple(out.shape)}")

    def _maybe_init_head(self, kind, out2d):
        # Build a Linear head only if we have embeddings
        if kind == "emb" and self.head is None:
            in_dim = out2d.shape[1]
            self.head = self.nn.Linear(in_dim, self.n_classes).to(self.device)

    def _logits(self, X):
        torch = self.torch
        self.backbone.eval()
        if self.head is not None:
            self.head.eval()
        empty_categ = torch.zeros((X.shape[0], 0), dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self._forward_backbone(X, empty_categ)
            kind, out2d = self._ensure_2d_features(out)
            return out2d if kind == "logits" else self.head(out2d)

    # ----- API -----
    def fit(self, X, y, sample_weight=None, X_valid=None, y_valid=None):
        torch = self.torch
        nn = self.nn
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.long, device=self.device)
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        Xv = yv = None
        if X_valid is not None:
            Xv = torch.tensor(X_valid, dtype=torch.float32, device=self.device)
            yv = torch.tensor(y_valid, dtype=torch.long, device=self.device)

        # ---- lazy head init: probe one batch
        empty_categ = torch.zeros((min(len(dataset), 4), 0), dtype=torch.long, device=self.device)
        with torch.no_grad():
            probe_out = self._forward_backbone(X[:empty_categ.shape[0]], empty_categ)
            kind, probe_2d = self._ensure_2d_features(probe_out)
        self._maybe_init_head(kind, probe_2d)

        # params to optimize
        params = list(self.backbone.parameters()) + (list(self.head.parameters()) if self.head is not None else [])
        opt = torch.optim.AdamW(params, lr=self.lr)

        # class weights (optional)
        if sample_weight is not None:
            # normalize to mean 1 for stability
            sw = torch.tensor(sample_weight, dtype=torch.float32, device=self.device)
            sw = sw / (sw.mean() + 1e-8)
        else:
            sw = None

        crit = FocalLoss(gamma=2.0)
        best_loss, bad = float("inf"), 0

        for epoch in range(self.max_epochs):
            self.backbone.train()
            if self.head is not None: self.head.train()
            for xb, yb in loader:
                opt.zero_grad()
                empty_categ = torch.zeros((xb.shape[0], 0), dtype=torch.long, device=self.device)
                out = self._forward_backbone(xb, empty_categ)
                kind, out2d = self._ensure_2d_features(out)
                logits = self.head(out2d) if kind == "emb" else out2d
                loss = crit(logits, yb)
                loss.backward()
                opt.step()

            if Xv is not None:
                self.backbone.eval()
                if self.head is not None: self.head.eval()
                with torch.no_grad():
                    empty_categ = torch.zeros((Xv.shape[0], 0), dtype=torch.long, device=self.device)
                    out = self._forward_backbone(Xv, empty_categ)
                    kind, out2d = self._ensure_2d_features(out)
                    logits = self.head(out2d) if kind == "emb" else out2d
                    vloss = nn.CrossEntropyLoss()(logits, yv).item()
                if vloss + 1e-9 < best_loss:
                    best_loss = vloss
                    # save best state for both backbone and head (if any)
                    self._best_state = {
                        "backbone": {k: v.clone() for k, v in self.backbone.state_dict().items()},
                        "head": ( {k: v.clone() for k, v in self.head.state_dict().items()} if self.head is not None else None )
                    }
                    bad = 0
                else:
                    bad += 1
                if self.verbose:
                    print(f"  [TabTransformer] epoch {epoch+1:03d} val_loss={vloss:.4f}")
                if bad >= self.patience:
                    break

        if self._best_state is not None:
            self.backbone.load_state_dict(self._best_state["backbone"])
            if self.head is not None and self._best_state["head"] is not None:
                self.head.load_state_dict(self._best_state["head"])

    def predict_proba(self, X):
        torch = self.torch
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        logits = self._logits(X)
        return torch.softmax(logits, dim=1).cpu().numpy()

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)

# Deterministic seeding for reproducibility
import random
random.seed(42)
try:
    import torch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    pass

# =========================================
# Config
# =========================================
INFILE = "processed/exo_unified.parquet"
OUT_MODELS  = Path("models");  OUT_MODELS.mkdir(exist_ok=True)
OUT_REPORTS = Path("reports"); OUT_REPORTS.mkdir(exist_ok=True)

TARGET_MAP = {0: "False Positive", 1: "Candidate", 2: "Confirmed"}
N_CLASSES  = 3
N_SPLITS   = 5

# Keep aligned with Step 1 / Step 2
PHYS_FEATURES = [
    "period","dur_h","depth_ppm","ror","rade_Re","a_over_rs","impact","ecc","incl_deg",
    "insol_S","eqt_K","teff_K","logg_cgs","feh_dex","rad_Rs","mass_Ms","age_Gyr",
    "ra_deg","dec_deg","pm_ra_masyr","pm_dec_masyr","dist_pc","mag_T","mag_Kepler",
    "depth_from_ror_ppm","depth_consistency","dur_frac","depth_snr","dur_snr","pm_tot",
    "ecc_flag","high_b",
    "period_err_mean","period_rel_err","period_missing",
    "t0_bjd_err_mean","t0_bjd_rel_err","t0_bjd_missing",
    "dur_h_err_mean","dur_h_rel_err","dur_h_missing",
    "depth_ppm_err_mean","depth_ppm_rel_err","depth_ppm_missing",
    "ror_err_mean","ror_rel_err","ror_missing",
    "a_over_rs_err_mean","a_over_rs_rel_err","a_over_rs_missing",
    "rho_star_cgs","a_AU","a_over_rs_phys","a_over_rs_consistency",
    "Rp_Re_from_ror","geom_transit_prob","Teq_phys_K",
    "dur_model_h","dur_consistency",
    "H_G","H_K","giant_flag","valley_flag","low_prob_flag","dur_outlier_flag"
]


# =========================================
# Utilities
# =========================================
def ensure_numpy(x):
    return x.values if isinstance(x, pd.Series) else (x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x))

def pr_auc_macro(y_true, proba):
    y_true = ensure_numpy(y_true); proba = ensure_numpy(proba)
    scores = []
    for c in range(N_CLASSES):
        y_bin = (y_true == c).astype(int)
        scores.append(average_precision_score(y_bin, proba[:, c]))
    return float(np.nanmean(scores)), {TARGET_MAP[i]: scores[i] for i in range(N_CLASSES)}

def roc_auc_ovo(y_true, proba):
    try: return float(roc_auc_score(y_true, proba, multi_class="ovo"))
    except Exception: return float("nan")

def plot_pr_curves(y_true, proba, title_prefix, out_prefix):
    y_true = ensure_numpy(y_true); proba = ensure_numpy(proba)
    for c in range(N_CLASSES):
        y_bin = (y_true == c).astype(int)
        p, r, _ = precision_recall_curve(y_bin, proba[:, c])
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(r, p)
        ax.set_title(f"{title_prefix} – PR ({TARGET_MAP[c]})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(f"{out_prefix}_pr_{c}.png", dpi=160); plt.close(fig)

def plot_confusion(y_true, y_pred, title, outfile):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels([TARGET_MAP[i] for i in range(N_CLASSES)], rotation=45, ha="right")
    ax.set_yticklabels([TARGET_MAP[i] for i in range(N_CLASSES)])
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    fig.tight_layout(); fig.savefig(outfile, dpi=160); plt.close(fig)

def reliability_bins(y_true_bin, y_score, n_bins=10):
    """Return (bin_confidence, bin_accuracy, counts) for a binary problem."""
    y_true_bin = np.asarray(y_true_bin).astype(int)
    y_score = np.asarray(y_score).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins+1)
    bin_conf, bin_acc, counts = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        sel = (y_score >= lo) & (y_score < hi) if i < n_bins-1 else (y_score >= lo) & (y_score <= hi)
        if sel.any():
            conf = y_score[sel].mean()
            acc  = y_true_bin[sel].mean()
            bin_conf.append(conf); bin_acc.append(acc); counts.append(sel.sum())
        else:
            bin_conf.append((lo+hi)/2); bin_acc.append(np.nan); counts.append(0)
    return np.array(bin_conf), np.array(bin_acc), np.array(counts)

def plot_reliability_multiclass(y_true, proba, out_png_prefix, n_bins=12):
    """One-vs-rest reliability per class + overall weighted ECE printed to console."""
    y_true = np.asarray(y_true).astype(int)
    proba  = np.asarray(proba).astype(float)
    overall_ece_num, overall_ece_den = 0.0, 0
    for c in range(N_CLASSES):
        y_bin = (y_true == c).astype(int)
        conf, acc, cnt = reliability_bins(y_bin, proba[:, c], n_bins=n_bins)
        # Expected Calibration Error (weighted L1)
        valid = ~np.isnan(acc)
        ece_num = np.sum(np.abs(acc[valid] - conf[valid]) * cnt[valid])
        ece_den = np.sum(cnt[valid]) if np.sum(cnt[valid]) > 0 else 1
        overall_ece_num += ece_num; overall_ece_den += ece_den

        fig, ax = plt.subplots(figsize=(4.5,4.0))
        ax.plot([0,1],[0,1], linestyle="--", alpha=0.5)
        ax.plot(conf[valid], acc[valid], marker="o")
        ax.set_title(f"Reliability: {TARGET_MAP[c]}")
        ax.set_xlabel("Confidence"); ax.set_ylabel("Empirical accuracy")
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(f"{out_png_prefix}_class{c}.png", dpi=160); plt.close(fig)

    overall_ece = overall_ece_num / max(1, overall_ece_den)
    print(f"📏 Overall (weighted) ECE ≈ {overall_ece:.4f} for {out_png_prefix}")

def per_mission_metrics(df, y_true, proba, title, out_json_path):
    """Report PR-AUC macro per mission and save to JSON."""
    missions = []
    for name, flag in [("Kepler", "is_kepler"), ("K2", "is_k2"), ("TESS", "is_tess")]:
        if flag in df.columns and df[flag].sum() > 0:
            mask = (df[flag] == 1).values
            if mask.any():
                pr_m, _ = pr_auc_macro(y_true[mask], proba[mask])
                missions.append((name, int(mask.sum()), float(pr_m)))
    # print nicely
    print(f"\n🔎 Per-mission OOF metrics: {title}")
    for name, rows, pr in missions:
        print(f"  {name:<6} rows={rows:<6}  PR-AUC(macro)={pr:.4f}")
    # save
    payload = [{"mission": m[0], "rows": m[1], "pr_auc_macro": m[2]} for m in missions]
    with open(out_json_path, "w") as f:
        json.dump(payload, f, indent=2)

def class_weight_dict(y):
    classes = np.array(sorted(np.unique(y)))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return dict(zip(classes, weights))

# Physics-informed monotone map (+1 increasing → more plausible transit; -1 decreasing)
# Adjust signs if your validation suggests otherwise.
def build_monotone_map(feature_cols):
    mono = {
        "depth_snr": +1,
        "geom_transit_prob": +1,
        "dur_consistency": -1,
        "a_over_rs_consistency": -1,
        "impact": -1,
        "low_prob_flag": -1,
        "dur_outlier_flag": -1,
        "depth_consistency": -1,   # larger mismatch -> less plausible
    }
    return [mono.get(c, 0) for c in feature_cols]

def xgb_monotone_string(feature_cols):
    sign = {+1:"+1", -1:"-1"}
    vec = build_monotone_map(feature_cols)
    s = "(" + ",".join(sign.get(v, "0") for v in vec) + ")"
    return s

def physics_weight_block(dfX):
    def safe_z(col):
        x = pd.to_numeric(dfX.get(col), errors="coerce")
        if x is None or len(x.dropna()) == 0:
            return np.zeros(len(dfX))
        mu, sd = np.nanmean(x), np.nanstd(x)
        if not np.isfinite(mu): mu = 0.0
        if not np.isfinite(sd) or sd < 1e-9: sd = 1.0
        return (x - mu) / sd

    dsnr = safe_z("depth_snr") if "depth_snr" in dfX.columns else np.zeros(len(dfX))
    gtp  = safe_z("geom_transit_prob") if "geom_transit_prob" in dfX.columns else np.zeros(len(dfX))
    dcon = safe_z("dur_consistency") if "dur_consistency" in dfX.columns else np.zeros(len(dfX))

    # Ensure numpy arrays and replace NaNs
    dsnr = np.nan_to_num(dsnr, nan=0.0)
    gtp  = np.nan_to_num(gtp,  nan=0.0)
    dcon = np.nan_to_num(dcon, nan=0.0)

    sig = lambda t: 1/(1+np.exp(-t))
    w = sig(dsnr) * sig(gtp) * sig(-dcon)
    w = np.clip(w, 0.5, 2.0)

    # Final guard
    w[~np.isfinite(w)] = 1.0
    return w

def make_cost_aware_decision(P, C):
    # P: (n,3) calibrated probabilities; C: (3,3) cost matrix true x pred
    # Choose pred that minimizes expected cost for each sample
    exp_cost = P @ C.T
    return np.argmin(exp_cost, axis=1)

def conformal_thresholds(oof_proba, y_true, alpha=0.1):
    """
    Split-conformal via OOF: per-class nonconformity s = 1 - p_true.
    Return class-wise thresholds q_c so that P(y in set) >= 1-alpha (approx).
    """
    y_true = np.asarray(y_true).astype(int)
    scores_by_class = {c: [] for c in range(N_CLASSES)}
    for i in range(len(y_true)):
        p = oof_proba[i, y_true[i]]
        scores_by_class[y_true[i]].append(1.0 - p)
    q = {}
    for c in range(N_CLASSES):
        sc = np.array(scores_by_class[c]) if len(scores_by_class[c]) else np.array([1.0])
        # finite-sample quantile
        k = int(np.ceil((len(sc)+1)*(1-alpha)))-1
        k = np.clip(k, 0, len(sc)-1)
        q[c] = float(np.sort(sc)[k])
    return q

def conformal_set(P_row, q):
    # Return set of classes whose 1 - p_c <= q_c
    S = [c for c in range(N_CLASSES) if (1.0 - P_row[c]) <= q[c]]
    return S

def physics_veto_adjust(proba, df_slice):
    # Basic sanity: ror < 1; b < 1 + ror ; if violated, damp Confirmed
    p = proba.copy()
    ror = pd.to_numeric(df_slice.get("ror"), errors="coerce")
    b   = pd.to_numeric(df_slice.get("impact"), errors="coerce")
    bad = np.zeros(len(p), dtype=bool)
    if ror is not None:
        bad |= (ror >= 1.0).fillna(False).to_numpy()
    if (ror is not None) and (b is not None):
        bad |= (b > (1.0 + ror.fillna(0))).fillna(False).to_numpy()

    if bad.any():
        p[bad, 2] *= 0.3  # damp "Confirmed"
        denom = p.sum(axis=1, keepdims=True)
        denom[denom==0] = 1.0
        p = p / denom
    return p

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
# ----- TabTransformer save/load helpers -----
def save_tabtransformer(wrapper, path):
    import torch
    payload = {
        "backbone": wrapper.backbone.state_dict(),
        "head": (wrapper.head.state_dict() if wrapper.head is not None else None),
        "n_features": wrapper.n_features,
        "n_classes": wrapper.n_classes,
        "device": wrapper.device,
        # store head input width to rebuild head cleanly
        "head_in_dim": (wrapper.head.in_features if wrapper.head is not None else None),
        # minimal hyperparams so we can reconstruct the backbone
        "hparams": {
            "hidden_dim": 96,
            "depth": 4,
            "heads": 8,
            "attn_dropout": 0.1,
            "ff_dropout": 0.1,
        }
    }
    torch.save(payload, path)

def load_tabtransformer(path):
    import torch
    ckpt = torch.load(path, map_location="cpu")
    tt = SkDeepTabWrapper(
        n_features=ckpt["n_features"],
        n_classes=ckpt["n_classes"],
        device="cpu",
        hidden_dim=ckpt["hparams"]["hidden_dim"],
        depth=ckpt["hparams"]["depth"],
        heads=ckpt["hparams"]["heads"],
        attn_dropout=ckpt["hparams"]["attn_dropout"],
        ff_dropout=ckpt["hparams"]["ff_dropout"],
    )
    tt.backbone.load_state_dict(ckpt["backbone"])
    # rebuild and load head if present
    if ckpt.get("head") is not None:
        in_dim = ckpt.get("head_in_dim")
        if in_dim is None:
            raise RuntimeError("TabTransformer checkpoint missing head_in_dim.")
        tt.head = tt.nn.Linear(in_dim, tt.n_classes).to(tt.device)
        tt.head.load_state_dict(ckpt["head"])
    return tt


# =========================================
# Load data
# =========================================
print("🔭 Loading", INFILE)
df = pd.read_parquet(INFILE)

if "label_int" not in df.columns:
    raise RuntimeError("Unified dataset missing 'label_int'. Please run Step 1 ETL.")

# Outer split (≈15% holdout, remainder for CV). Group-aware by star_id when available.
from sklearn.model_selection import train_test_split
try:
    from sklearn.model_selection import StratifiedGroupKFold as _SGKF  # noqa
    HAS_SGKF = True
except Exception:
    HAS_SGKF = False

y_all = df["label_int"].astype(int).to_numpy()
groups_all = df["star_id"].astype(str).to_numpy() if "star_id" in df.columns else None

if groups_all is not None and HAS_SGKF:
    splitter_outer = _SGKF(n_splits=7, shuffle=True, random_state=42)  # ~15% holdout (1/7)
    train_idx, test_idx = next(splitter_outer.split(df, y_all, groups=groups_all))
    X_trainval = df.iloc[train_idx].reset_index(drop=True)
    X_test     = df.iloc[test_idx].reset_index(drop=True)
    y_trainval = y_all[train_idx]
    y_test     = y_all[test_idx]
else:
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        df, y_all, test_size=0.15, stratify=y_all, random_state=42
    )

print(f"🌍 Outer split: Train/Val = {len(X_trainval)}, Test = {len(X_test)}")
# Check class balance in Train/Val vs Test
print("\n📊 Class balance check:")
for name, y_part in [("Train/Val", y_trainval), ("Test", y_test)]:
    unique, counts = np.unique(y_part, return_counts=True)
    dist = {TARGET_MAP[int(u)]: int(c) for u, c in zip(unique, counts)}
    print(f"  {name:<9} -> {dist}")
# Per-mission balance diagnostics
for name, X_part, y_part in [("Train/Val", X_trainval, y_trainval), ("Test", X_test, y_test)]:
    print(f"\n🔎 Per-mission balance ({name}):")
    for mission, flag in [("Kepler", "is_kepler"), ("K2", "is_k2"), ("TESS", "is_tess")]:
        if flag in X_part.columns and X_part[flag].sum() > 0:
            mask = (X_part[flag] == 1).values
            if mask.any():
                unique, counts = np.unique(y_part[mask], return_counts=True)
                dist = {TARGET_MAP[int(u)]: int(c) for u, c in zip(unique, counts)}
                print(f"  {mission:<6} rows={mask.sum():<6} -> {dist}")

# For the rest of training, work only on Train/Val
df = X_trainval
y  = y_trainval
groups = df["star_id"].astype(str).to_numpy() if "star_id" in df.columns else None

# Feature selection: intersection + drop >98% NaN
feature_cols = [c for c in PHYS_FEATURES if c in df.columns]
keep_cols = []
for c in feature_cols:
    if df[c].isna().mean() < 0.98:
        keep_cols.append(c)
feature_cols = keep_cols

Xdf = df[feature_cols].copy()

# Baseline comparator (BLS S/N proxy) for benchmarking
if "depth_ppm" in Xdf.columns and "depth_ppm_err_mean" in Xdf.columns:
    denom = Xdf["depth_ppm_err_mean"].replace([0, np.inf, -np.inf], np.nan)
    Xdf["bls_sn_proxy"] = Xdf["depth_ppm"] / denom
    Xdf["bls_sn_proxy"].replace([np.inf, -np.inf], np.nan, inplace=True)
    Xdf["bls_sn_proxy"] = Xdf["bls_sn_proxy"].fillna(0.0)
    feature_cols.append("bls_sn_proxy")

# 🔧 Load medians from Step 4 if available, else fall back to dataset medians
meta_file = Path("processed/exo_unified.meta.json")
if meta_file.exists():
    medians = json.loads(meta_file.read_text()).get("medians", {})
    print(f"📐 Loaded {len(medians)} medians from {meta_file}")
else:
    print("⚠️ No exo_unified.meta.json found, computing medians from current dataset.")
    medians = {c: Xdf[c].median() for c in feature_cols}

for c in feature_cols:
    Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce").fillna(
        medians.get(c, np.nanmedian(Xdf[c]))
    )

X = Xdf.to_numpy().astype(float)

print(f"✅ Features: {len(feature_cols)} columns")
print("Some features:", feature_cols[:20], "...")

# =========================================
# Mission re-weighting (domain shift → TESS-like importance)
# =========================================
is_tess = df["is_tess"].fillna(0).astype(int).to_numpy() if "is_tess" in df.columns else np.zeros(len(y), dtype=int)
dr_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
dr_clf.fit(X, is_tess)
mission_w_all = 0.5 + dr_clf.predict_proba(X)[:,1]  # ~[0.5, 1.5]

# =========================================
# CV split
# =========================================
if HAS_SGKF and groups is not None:
    splitter = _SGKF(n_splits=N_SPLITS, shuffle=True, random_state=42)
    cv_iter = splitter.split(X, y, groups=groups)
else:
    print("⚠️ Using StratifiedKFold fallback.")
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    cv_iter = splitter.split(X, y)

# =========================================
# Base models with physics-aware constraints
# =========================================
BASE_MODELS = {}
if HAS_LGBM:
    BASE_MODELS["lgbm"] = LGBMClassifier(
        n_estimators=900,
        learning_rate=0.03,
        num_leaves=63,
        objective="multiclass",
        class_weight="balanced",
        random_state=42,
        # imbalance + subsampling knobs
        is_unbalance=True,
        subsample_freq=1,
        subsample=0.8,
        monotone_constraints=build_monotone_map(feature_cols)
    )
    # Optional: TabTransformer (if available) or TabNet fallback
try:
    if HAS_TABTRANS:
        device = "cuda" if ("torch" in globals() and torch.cuda.is_available()) else "cpu"
        BASE_MODELS["tabtransformer"] = ("deep", dict(device=device))
    elif HAS_TABNET:
        BASE_MODELS["tabnet"] = ("tabnet", dict())  # params can be tuned
except Exception as _e:
    pass

if HAS_XGB:
    BASE_MODELS["xgb"] = XGBClassifier(
        n_estimators=900, learning_rate=0.03, max_depth=7,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=N_CLASSES, random_state=42,
        tree_method="hist",
        monotone_constraints=xgb_monotone_string(feature_cols)
    )
if HAS_CAT:
    BASE_MODELS["cat"] = CatBoostClassifier(
        loss_function="MultiClass", iterations=1200, depth=8, learning_rate=0.03,
        l2_leaf_reg=3.0, random_seed=42, verbose=False
    )

if not BASE_MODELS:
    raise RuntimeError("No boosting library available. Install lightgbm / xgboost / catboost.")

# =========================================
# CV training with OOF
# =========================================
oof_proba = {k: np.zeros((len(y), N_CLASSES)) for k in BASE_MODELS}
fold_scores = defaultdict(list)
models_per_fold = defaultdict(list)

global_cls_w = class_weight_dict(y)

for fold, (tr_idx, va_idx) in enumerate(cv_iter, start=1):
    print(f"\n🧪 Fold {fold}/{N_SPLITS}")
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]

    # Optional oversampling (per-fold, to balance class counts)
    unique, counts = np.unique(ytr, return_counts=True)
    min_count = counts.min()
    max_count = counts.max()
    if max_count / min_count > 1.5:  # imbalance threshold
        print(f"   ↪ Oversampling minority classes (fold {fold})...")
        Xtr_bal, ytr_bal = [], []
        for c in np.unique(ytr):
            mask = (ytr == c)
            X_c = Xtr[mask]; y_c = ytr[mask]
            X_res, y_res = resample(
                X_c, y_c,
                replace=True,
                n_samples=max_count,
                random_state=42
            )
            Xtr_bal.append(X_res); ytr_bal.append(y_res)
        Xtr = np.vstack(Xtr_bal)
        ytr = np.concatenate(ytr_bal)

    # Per-fold weights: class * physics * mission
    # (re)build DataFrame after potential oversampling
    Xtr_df = pd.DataFrame(Xtr, columns=feature_cols)

    # physics weights
    phys_w_tr = physics_weight_block(Xtr_df)

    # mission weights must align with (possibly oversampled) Xtr
    mission_w_tr = 0.5 + dr_clf.predict_proba(Xtr)[:, 1]  # domain ratio model

    # class weights
    cw = class_weight_dict(ytr)
    class_w_tr = np.array([cw[c] for c in ytr])

    # final sample weights
    sample_w = class_w_tr * phys_w_tr * mission_w_tr

    for key, model in BASE_MODELS.items():
        print(f"  ▶ Training {key.upper()}...")
        if key == "cat":
            cw_vec = [cw.get(i, 1.0) for i in range(N_CLASSES)]
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(
                loss_function="MultiClass", iterations=1200, depth=8,
                learning_rate=0.03, l2_leaf_reg=3.0, random_seed=42, verbose=False,
                class_weights=cw_vec
            )
            model.fit(Xtr, ytr, sample_weight=sample_w, eval_set=(Xva, yva), verbose=False)
            proba = model.predict_proba(Xva)

        elif key == "tabtransformer" and isinstance(model, tuple) and model[0] == "deep":
            params = model[1]
            tt = SkDeepTabWrapper(
                n_features=Xtr.shape[1], n_classes=N_CLASSES,
                device=params.get("device","cpu"),
                hidden_dim=96, depth=4, heads=8, attn_dropout=0.1, ff_dropout=0.1,
                lr=2e-3, batch_size=1024, max_epochs=200, patience=25
            )
            tt.fit(Xtr, ytr, X_valid=Xva, y_valid=yva)
            proba = tt.predict_proba(Xva)
            model = tt  # keep fitted object

        elif key == "tabnet" and isinstance(model, tuple) and model[0] == "tabnet":
            # TabNet needs numpy and accepts sample_weight
            tn = TabNetClassifier(  # basic params; tune if you like
                n_d=64, n_a=64, n_steps=5, gamma=1.5,
                n_independent=2, n_shared=2, momentum=0.02,
                lambda_sparse=1e-4, seed=42, verbose=0
            )
            tn.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_name=["valid"], max_epochs=200,
                patience=30, batch_size=2048, virtual_batch_size=128, weights=sample_w)
            proba = tn.predict_proba(Xva)
            model = tn

        else:
            model.fit(Xtr, ytr, sample_weight=sample_w)
            proba = model.predict_proba(Xva)

        # Physics veto adjustment on validation predictions
        proba = physics_veto_adjust(proba, pd.DataFrame(Xva, columns=feature_cols))

        oof_proba[key][va_idx] = proba
        models_per_fold[key].append(model)

        pr_macro, _ = pr_auc_macro(yva, proba)
        roc_ovo = roc_auc_ovo(yva, proba)
        fold_scores[(key,"pr_auc_macro")].append(pr_macro)
        fold_scores[(key,"roc_auc_ovo")].append(roc_ovo)
        print(f"    PR-AUC (macro): {pr_macro:.4f} | ROC-AUC (ovo): {roc_ovo:.4f}")

# =========================================
# OOF summary + ensemble weights
# =========================================
base_report = {}; weights = {}
for key in BASE_MODELS:
    pr_macro, per_class = pr_auc_macro(y, oof_proba[key])
    roc_ovo = roc_auc_ovo(y, oof_proba[key])
    base_report[key] = {
        "oof_pr_auc_macro": pr_macro,
        "per_class": per_class,
        "oof_roc_auc_ovo": roc_ovo,
        "cv_pr_mean": float(np.mean(fold_scores[(key,"pr_auc_macro")])),
        "cv_pr_std":  float(np.std(fold_scores[(key,"pr_auc_macro")])),
    }
    weights[key] = pr_macro ** 2

# Normalize weights
w_sum = sum(weights.values())
weights = {k: (v / w_sum if w_sum>0 else 1.0/len(weights)) for k,v in weights.items()}

print("\n🔗 Base-model OOF summary:")
for k,v in base_report.items():
    print(f"  {k}: PR-AUC(macro)={v['oof_pr_auc_macro']:.4f}  ROC-AUC(ovo)={v['oof_roc_auc_ovo']:.4f}  w={weights[k]:.3f}")

# =========================================
# Stacking + calibration
# =========================================
# Weighted average (physics veto already applied in OOF)
oof_weighted = np.zeros((len(y), N_CLASSES))
for k in BASE_MODELS:
    oof_weighted += weights[k] * oof_proba[k]

# Meta on concatenated probs
stack_matrix = np.hstack([oof_proba[k] for k in BASE_MODELS])
meta = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", random_state=42)
meta.fit(stack_matrix, y)
oof_meta = meta.predict_proba(stack_matrix)

# Calibrate weighted avg
cal_meta = CalibratedClassifierCV(
    estimator=OneVsRestClassifier(LogisticRegression(max_iter=1000)),
    method="isotonic", cv=3
)
cal_meta.fit(oof_weighted, y)
oof_cal = cal_meta.predict_proba(oof_weighted)

pr_w, _ = pr_auc_macro(y, oof_weighted)
pr_meta, _ = pr_auc_macro(y, oof_meta)
pr_cal, _ = pr_auc_macro(y, oof_cal)
print(f"\n🧮 OOF PR-AUC (weighted): {pr_w:.4f}")
print(f"🧮 OOF PR-AUC (stack meta): {pr_meta:.4f}")
print(f"🧮 OOF PR-AUC (calibrated weighted): {pr_cal:.4f}")
# =========================================
# Optimal decision thresholds (per-class)
# =========================================
from sklearn.metrics import f1_score

best_thresh = {}
y_pred_thresh = np.zeros_like(y)

for c in range(N_CLASSES):
    y_bin = (y == c).astype(int)
    prec, rec, th = precision_recall_curve(y_bin, oof_cal[:, c])
    f1s = 2 * (prec * rec) / (prec + rec + 1e-9)
    idx = np.nanargmax(f1s)
    best_thresh[c] = float(th[idx]) if idx < len(th) else 0.5

print("\n🎯 Learned per-class thresholds:")
for c in range(N_CLASSES):
    print(f"  {TARGET_MAP[c]}: {best_thresh[c]:.3f}")
# Per-mission diagnostics
per_mission_metrics(df, y, oof_weighted, "Weighted ensemble", OUT_REPORTS/"step3_perm_mission_weighted.json")
per_mission_metrics(df, y, oof_cal,      "Calibrated ensemble", OUT_REPORTS/"step3_perm_mission_calibrated.json")


# =========================================
# Conformal thresholds (optional but powerful)
# =========================================
q_thresh = conformal_thresholds(oof_cal, y, alpha=0.10)  # 90% target coverage

# =========================================
# Reports
# =========================================
plot_pr_curves(y, oof_weighted, "OOF Weighted Ensemble", str(OUT_REPORTS/"step3_weighted"))
plot_pr_curves(y, oof_cal,      "OOF Calibrated Ensemble", str(OUT_REPORTS/"step3_calibrated"))
# Reliability diagrams (one-vs-rest per class + overall ECE)
plot_reliability_multiclass(y, oof_weighted, str(OUT_REPORTS/"step3_reliability_weighted"))
plot_reliability_multiclass(y, oof_cal,      str(OUT_REPORTS/"step3_reliability_calibrated"))


y_pred_weighted = np.argmax(oof_weighted, axis=1)
plot_confusion(y, y_pred_weighted, "OOF Confusion – Weighted", OUT_REPORTS/"step3_conf_weighted.png")

y_pred_cal = np.argmax(oof_cal, axis=1)
plot_confusion(y, y_pred_cal, "OOF Confusion – Calibrated", OUT_REPORTS/"step3_conf_calibrated.png")

def predict_with_thresholds(proba, thresh_dict):
    proba = np.asarray(proba)
    margins = np.array([proba[:, c] - thresh_dict.get(c, 0.5) for c in range(proba.shape[1])]).T
    chosen = np.argmax(margins, axis=1)
    # fallback to argmax if no class beats its threshold
    beats = (margins >= 0).any(axis=1)
    argmax_plain = np.argmax(proba, axis=1)
    return np.where(beats, chosen, argmax_plain)

y_pred_thresh = predict_with_thresholds(oof_cal, best_thresh)
plot_confusion(y, y_pred_thresh, "OOF Confusion – Thresholded", OUT_REPORTS/"step3_conf_thresholded.png")

# Cost-aware decision (example costs; tune with science team)
COST = np.array([
    [0.0, 0.5, 2.0],  # true FP → pred [FP, Cand, Conf]
    [0.5, 0.0, 1.0],  # true Cand
    [1.0, 0.5, 0.0],  # true Conf
])
y_costaware = make_cost_aware_decision(oof_cal, COST)
plot_confusion(y, y_costaware, "OOF Confusion – Cost-aware", OUT_REPORTS/"step3_conf_costaware.png")


# =========================================
# Refit on ALL data
# =========================================
cls_w_all = class_weight_dict(y)
class_w_vec = np.array([cls_w_all[c] for c in y])
phys_w_all  = physics_weight_block(pd.DataFrame(X, columns=feature_cols))
sample_w_all = class_w_vec * phys_w_all * mission_w_all

# Optional: inverse-uncertainty weighting (Malik 2022)
unc_cols = [c for c in ["period_rel_err","depth_ppm_rel_err","dur_h_rel_err"] if c in Xdf.columns]
if unc_cols:
    unc_mean = Xdf[unc_cols].mean(axis=1).to_numpy()
    unc_w = 1.0 / (1.0 + np.nan_to_num(unc_mean, nan=0.0))
    sample_w_all = sample_w_all * unc_w

fitted = {}
for key, base in BASE_MODELS.items():
    print(f"\n🧩 Refit on all data: {key}")
    if key == "cat":
        cw_vec = [cls_w_all.get(i,1.0) for i in range(N_CLASSES)]
        from catboost import CatBoostClassifier
        cat = CatBoostClassifier(
            loss_function="MultiClass", iterations=1200, depth=8,
            learning_rate=0.03, l2_leaf_reg=3.0, random_seed=42, verbose=False,
            class_weights=cw_vec
        )
        cat.fit(X, y, sample_weight=sample_w_all, verbose=False)
        fitted[key] = cat

    elif key == "tabtransformer" and isinstance(base, tuple) and base[0] == "deep":
        params = base[1]
        tt = SkDeepTabWrapper(
            n_features=X.shape[1], n_classes=N_CLASSES,
            device=params.get("device","cpu"),
            hidden_dim=96, depth=4, heads=8, attn_dropout=0.1, ff_dropout=0.1,
            lr=2e-3, batch_size=1024, max_epochs=300, patience=30
        )
        tt.fit(X, y)  # fit on all data
        fitted[key] = tt

    elif key == "tabnet" and isinstance(base, tuple) and base[0] == "tabnet":
        tn = TabNetClassifier(
            n_d=64, n_a=64, n_steps=5, gamma=1.5,
            n_independent=2, n_shared=2, momentum=0.02,
            lambda_sparse=1e-4, seed=42, verbose=0
        )
        tn.fit(X, y, max_epochs=300, patience=30, batch_size=4096, virtual_batch_size=256, weights=sample_w_all)
        fitted[key] = tn

    else:
        base.fit(X, y, sample_weight=sample_w_all)
        fitted[key] = base

# Deploy-time stacked + calibrated models
P_full_concat = np.hstack([fitted[k].predict_proba(X) for k in fitted])
meta_deploy = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", random_state=42)
meta_deploy.fit(P_full_concat, y)

P_full_weighted = np.zeros((len(y), N_CLASSES))
for k in fitted:
    P_full_weighted += weights[k] * fitted[k].predict_proba(X)
P_full_weighted = physics_veto_adjust(P_full_weighted, pd.DataFrame(X, columns=feature_cols))

cal_meta_deploy = CalibratedClassifierCV(
    estimator=OneVsRestClassifier(LogisticRegression(max_iter=1000)),
    method="isotonic", cv=3
)
cal_meta_deploy.fit(P_full_weighted, y)

# Save artifacts
artifacts = {
    "feature_cols": feature_cols,
    "medians": {c: float(m) if pd.notna(m) else None for c,m in pd.Series(medians).items()},
    "weights": {k: float(v) for k,v in weights.items()},
    "target_map": TARGET_MAP,
    "base_report": base_report,
    "scores": {
        "oof": {
            "weighted_pr_auc_macro": pr_w,
            "stack_meta_pr_auc_macro": pr_meta,
            "calibrated_pr_auc_macro": pr_cal
        }
    },
    "conformal_q": q_thresh,
    "cost_matrix": COST.tolist(),
    "best_thresholds": best_thresh
}
# =========================================
# Final NASA Demo Hold-Out Test Evaluation
# =========================================
print("\n🛰️ Evaluating on NASA demo hold-out test set (15%)...")

# Prepare features for test set (rebuild engineered columns if needed)
Xdf_test = X_test.copy()

# Rebuild bls_sn_proxy if possible
if "depth_ppm" in Xdf_test.columns and "depth_ppm_err_mean" in Xdf_test.columns:
    denom = Xdf_test["depth_ppm_err_mean"].replace([0, np.inf, -np.inf], np.nan)
    Xdf_test["bls_sn_proxy"] = (
        Xdf_test["depth_ppm"] / denom
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

# Ensure all expected features exist, fill missing ones with 0
for c in feature_cols:
    if c not in Xdf_test.columns:
        Xdf_test[c] = 0.0

# Reorder to feature_cols and fill NaNs with medians
Xdf_test = Xdf_test[feature_cols].copy()
for c in feature_cols:
    Xdf_test[c] = pd.to_numeric(Xdf_test[c], errors="coerce").fillna(
        medians.get(c, np.nanmedian(Xdf_test[c]))
    )
X_test_np = Xdf_test.to_numpy().astype(float)

# Get calibrated ensemble predictions
P_weighted_test = np.zeros((len(X_test_np), N_CLASSES))
for k in fitted:
    Pk = fitted[k].predict_proba(X_test_np)
    P_weighted_test += weights[k] * Pk
P_weighted_test = physics_veto_adjust(P_weighted_test, Xdf_test)
P_cal_test = cal_meta_deploy.predict_proba(P_weighted_test)

# Metrics
pr_test, _ = pr_auc_macro(y_test, P_cal_test)
roc_test = roc_auc_ovo(y_test, P_cal_test)
print(f"🧪 Hold-out PR-AUC (macro): {pr_test:.4f} | ROC-AUC (ovo): {roc_test:.4f}")

plot_pr_curves(y_test, P_cal_test, "Hold-out Calibrated Ensemble", str(OUT_REPORTS/"step3_holdout"))
plot_confusion(y_test, np.argmax(P_cal_test, axis=1), "Hold-out Confusion", OUT_REPORTS/"step3_conf_holdout.png")

# ➕ Add hold-out metrics to artifacts
artifacts["scores"]["holdout"] = {
    "pr_auc_macro": float(pr_test),
    "roc_auc_ovo":  float(roc_test)
}
import platform, hashlib, sklearn
try: import xgboost
except: xgboost = None
try: import lightgbm
except: lightgbm = None
try: import catboost
except: catboost = None

data_hash = hashlib.md5(open(INFILE,"rb").read()).hexdigest()
artifacts["reproducibility"] = {
    "random_seed": 42,
    "python_version": platform.python_version(),
    "sklearn": sklearn.__version__,
    "xgboost": getattr(xgboost,"__version__", None),
    "lightgbm": getattr(lightgbm,"__version__", None),
    "catboost": getattr(catboost,"__version__", None),
    "dataset_hash": data_hash
}

# Save updated report JSON
save_json(OUT_REPORTS/"step3_report.json", artifacts)

for k, m in fitted.items():
    if k == "tabtransformer":
        # use our helper to store backbone+head+meta
        save_tabtransformer(m, OUT_MODELS / "step3_tabtransformer.pt")
    elif k == "tabnet":
        try:
            # TabNet has its own saver; more robust than joblib
            m.save_model(str(OUT_MODELS / "step3_tabnet.zip"))
        except Exception:
            # fallback to joblib if available
            joblib.dump(m, OUT_MODELS / "step3_tabnet.pkl")
    else:
        joblib.dump(m, OUT_MODELS / f"step3_{k}.pkl")
joblib.dump(meta_deploy, OUT_MODELS/"step3_stack_meta.pkl")
joblib.dump(cal_meta_deploy, OUT_MODELS/"step3_cal_meta.pkl")
joblib.dump(dr_clf, OUT_MODELS/"step3_domain_ratio.pkl")
save_json(OUT_MODELS/"step3_features.json", {
    "feature_cols": feature_cols,
    "medians": artifacts["medians"],
    "weights": artifacts["weights"],
    "target_map": TARGET_MAP
})

print("\n✅ Step 3 complete. Models & reports saved.")

# =========================================
# Optional: SHAP global importance (LightGBM if available)
# =========================================
try:
    if HAS_SHAP and "lgbm" in fitted:
        explainer = shap.TreeExplainer(fitted["lgbm"])
        shap_vals = explainer.shap_values(X, check_additivity=False)

        # Normalize format (list vs array)
        if isinstance(shap_vals, list):
            shap_class_vals = shap_vals[2]   # "Confirmed" class
        else:
            shap_class_vals = shap_vals

        # Summary plot
        shap.summary_plot(shap_class_vals, X, feature_names=feature_cols, show=False)
        plt.tight_layout(); plt.savefig(OUT_REPORTS/"step3_shap_summary_confirmed.png", dpi=160); plt.close()

        # Global bar plot
        shap.summary_plot(shap_class_vals, X, feature_names=feature_cols,
                          plot_type="bar", show=False)
        plt.tight_layout(); plt.savefig(OUT_REPORTS/"step3_shap_bar_confirmed.png", dpi=160); plt.close()
        print("🖼️ SHAP plots saved.")
except Exception as e:
    print(f"[SHAP] Skipped ({e})")

# =========================================
# Inference helper (optional): cost-aware + conformal + veto
# =========================================
def load_step3_bundle():
    feats = json.load(open(OUT_MODELS / "step3_features.json"))
    model_names = list(feats["weights"].keys())  # e.g., ["lgbm", "xgb", "cat", "tabtransformer", ...]
    models = {}
    for k in model_names:
        if k == "tabtransformer":
            models[k] = load_tabtransformer(OUT_MODELS / "step3_tabtransformer.pt")
        elif k == "tabnet":
            # prefer native loader if zip exists; fallback to joblib
            zip_path = OUT_MODELS / "step3_tabnet.zip"
            pkl_path = OUT_MODELS / "step3_tabnet.pkl"
            if zip_path.exists():
                tn = TabNetClassifier()
                tn.load_model(str(zip_path))
                models[k] = tn
            else:
                models[k] = joblib.load(pkl_path)
        else:
            models[k] = joblib.load(OUT_MODELS / f"step3_{k}.pkl")
    # NOTE: meta (stacking logistic regression) is saved for reference only,
    # inference uses calibrated weighted ensemble (cal).

    meta = joblib.load(OUT_MODELS / "step3_stack_meta.pkl")
    cal  = joblib.load(OUT_MODELS / "step3_cal_meta.pkl")
    dr   = joblib.load(OUT_MODELS / "step3_domain_ratio.pkl")
    q    = json.load(open(OUT_REPORTS / "step3_report.json"))["conformal_q"]
    return feats, models, meta, cal, dr, q

def predict_step3(df_in):
    """df_in: DataFrame with raw columns superset; returns dict with proba, decisions, sets."""
    feats, models, meta, cal, dr, q = load_step3_bundle()
    feature_cols = feats["feature_cols"]; medians = feats["medians"]; weights = feats["weights"]

    Xdf = df_in.reindex(columns=feature_cols)
    for c in feature_cols:
        Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce").fillna(medians[c])
    X = Xdf.to_numpy().astype(float)

    # base probs
    P_weighted = np.zeros((len(X), N_CLASSES))
    for k in models:
        Pk = models[k].predict_proba(X)
        P_weighted += weights[k] * Pk

    # physics veto adjust
    P_weighted = physics_veto_adjust(P_weighted, Xdf)

    # calibration
    P_cal = cal.predict_proba(P_weighted)

    # cost-aware decision
    C = np.array(json.load(open(OUT_REPORTS/"step3_report.json"))["cost_matrix"])
    y_costaware = make_cost_aware_decision(P_cal, C)

    # conformal sets
    sets = [conformal_set(P_cal[i], q) for i in range(len(X))]

    return {
        "proba_calibrated": P_cal,
        "pred_costaware": y_costaware,
        "prediction_sets": sets
    }

if __name__ == "__main__":
    print("Saved:")
    print("  - reports/step3_report.json")
    print("  - reports/step3_* PR curves & confusion matrices")
    print("  - models/step3_*.pkl + meta/calibration/domain_ratio + features")
