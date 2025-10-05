# step2_train.py
import os, json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from astropy.timeseries import BoxLeastSquares
import platform, hashlib
import sklearn, xgboost, lightgbm, catboost
import time

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, average_precision_score,
                             precision_recall_curve, roc_curve)
from sklearn.model_selection import StratifiedKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except Exception:
    HAS_SGKF = False
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional model libs (we’ll skip gracefully if missing)
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
    from catboost import CatBoostClassifier, Pool
except Exception:
    HAS_CAT = False

import joblib
from sklearn.multiclass import OneVsRestClassifier

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)

# =========================================================
# Config
# =========================================================
INFILE = "processed/exo_unified.parquet"  # from Step 1
OUT_MODELS = Path("models"); OUT_MODELS.mkdir(exist_ok=True)
OUT_REPORTS = Path("reports"); OUT_REPORTS.mkdir(exist_ok=True)

TARGET_NAME = "label_int"   # 0: False Positive, 1: Candidate, 2: Confirmed
TARGET_MAP  = {0: "False Positive", 1: "Candidate", 2: "Confirmed"}
N_CLASSES   = 3
N_SPLITS    = 5

# Core feature whitelist (we’ll take intersection with actual columns)
PHYS_FEATURES = [
    # transit & geometry
    "period","dur_h","depth_ppm","ror","rade_Re","a_over_rs","impact","ecc","incl_deg",
    # stellar & irradiation
    "insol_S","eqt_K","teff_K","logg_cgs","feh_dex","rad_Rs","mass_Ms","age_Gyr",
    # astrometry & mags (safe)
    "ra_deg","dec_deg","pm_ra_masyr","pm_dec_masyr","dist_pc","mag_T","mag_Kepler",
    # engineered (Step 1)
    "depth_from_ror_ppm","depth_consistency","dur_frac","depth_snr","dur_snr","pm_tot",
    "ecc_flag","high_b",
    # uncertainty bundles (Step 1 auto-generated if errors present)
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

META_DROP = ["rowid","label_raw","label","star_id","is_kepler","is_k2","is_tess"]  # keep mission flags for per-mission eval only

# Reasonable default hyperparameters (physics-friendly, conservative)
PARAM_XGB = dict(
    n_estimators=600, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective="multi:softprob", num_class=N_CLASSES,
    tree_method="hist", reg_lambda=1.0, reg_alpha=0.0, random_state=42
)
PARAM_LGBM = dict(
    n_estimators=800, num_leaves=63, max_depth=-1, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    objective="multiclass", class_weight="balanced",
    reg_lambda=1.0, random_state=42
)
PARAM_CAT = dict(
    loss_function="MultiClass", iterations=1200, depth=8, learning_rate=0.03,
    l2_leaf_reg=3.0, random_seed=42, allow_writing_files=False, verbose=False
)

# =========================================================
# Utilities
# =========================================================
def ensure_numpy(x):
    return x.values if isinstance(x, pd.Series) else (x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x))

def pr_auc_macro(y_true, proba):
    # macro-average PR-AUC (one-vs-rest)
    scores = []
    y_true = ensure_numpy(y_true)
    proba  = ensure_numpy(proba)
    for c in range(N_CLASSES):
        y_bin = (y_true == c).astype(int)
        scores.append(average_precision_score(y_bin, proba[:, c]))
    return float(np.nanmean(scores)), dict(zip([TARGET_MAP[i] for i in range(N_CLASSES)], scores))

def roc_auc_ovo(y_true, proba):
    try:
        return float(roc_auc_score(y_true, proba, multi_class="ovo"))
    except Exception:
        return float("nan")

def plot_confusion(y_true, y_pred, title, outfile):
    cm = confusion_matrix(y_true, y_pred, labels=list(TARGET_MAP.keys()), normalize=None)
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

def plot_pr_curves(y_true, proba, title_prefix, out_prefix):
    y_true = ensure_numpy(y_true); proba = ensure_numpy(proba)
    for c in range(N_CLASSES):
        y_bin = (y_true == c).astype(int)
        p, r, _ = precision_recall_curve(y_bin, proba[:, c])
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(r, p)
        ax.set_title(f"{title_prefix} – PR curve ({TARGET_MAP[c]})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(f"{out_prefix}_pr_{c}.png", dpi=160); plt.close(fig)

def class_weight_vector(y):
    # sklearn expects dict {class: weight}
    classes = np.array(sorted(np.unique(y)))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return dict(zip(classes, weights))

def gather_features(df):
    # select intersection of PHYS_FEATURES with actual columns, drop high-NA columns
    cols = [c for c in PHYS_FEATURES if c in df.columns]
    # remove columns that are >98% NaN to keep models stable across missions
    keep = []
    for c in cols:
        na_frac = df[c].isna().mean()
        if na_frac < 0.98:
            keep.append(c)
    return keep

# =========================================================
# Load data
# =========================================================
print("🔬 Loading unified dataset from", INFILE)
df = pd.read_parquet(INFILE)

# Mission flags for per-mission eval; not in the feature matrix
has_flag_cols = [c for c in ["is_kepler","is_k2","is_tess"] if c in df.columns]
for c in has_flag_cols:
    df[c] = df[c].fillna(0).astype(int)

# Safety: enforce correct target dtype
if "label_int" not in df.columns:
    raise RuntimeError("Unified dataset missing 'label_int'. Re-run Step 1.")
df = df.dropna(subset=["label_int"])
y = df["label_int"].astype(int).to_numpy()
groups = df["star_id"].astype(str).to_numpy()
# ---------- Advanced physics augmentation (insert before gather_features) ----------
G_SI = 6.67430e-11
R_SUN = 6.957e8
M_SUN = 1.98847e30
AU_M  = 1.495978707e11
DAY_S = 86400.0
SIGMA = 5.670374419e-8

def safe_log10(x):
    x = pd.to_numeric(x, errors="coerce")
    return np.log10(np.where(x>0, x, np.nan))

# Stellar mass/radius/density in SI where possible
rad_Rs = pd.to_numeric(df.get("rad_Rs"), errors="coerce")                  # R*/R☉
mass_Ms = pd.to_numeric(df.get("mass_Ms"), errors="coerce")                # M*/M☉
logg = pd.to_numeric(df.get("logg_cgs"), errors="coerce")                  # cgs

# Estimate M* from logg and R* if missing (g = GM/R^2)
g_cgs = 10**logg if logg.notna().any() else pd.Series(index=df.index, dtype=float)
M_from_logg = (g_cgs * (rad_Rs*R_SUN)**2) / G_SI / M_SUN  # in M☉ (since M_SUN divides)
mass_est = mass_Ms.copy()
mass_est = mass_est.where(mass_est.notna(), M_from_logg)

# Stellar density from M* and R*
rho_star = (3.0 * (mass_est*M_SUN)) / (4.0*np.pi * (rad_Rs*R_SUN)**3)     # kg/m^3
df["rho_star_cgs"] = rho_star * 1e-3  # kg/m^3 -> g/cm^3

# Semi-major axis from Kepler's 3rd law (assuming M_p << M*)
P_days = pd.to_numeric(df.get("period"), errors="coerce")
P_sec  = P_days * DAY_S
a_m = (G_SI * (mass_est*M_SUN) * (P_sec**2) / (4*np.pi**2)) ** (1/3)
df["a_AU"] = a_m / AU_M

# a/R* (from period and rho_star) and consistency with catalog a_over_rs (if present)
a_over_rs_phys = a_m / (rad_Rs*R_SUN)
df["a_over_rs_phys"] = a_over_rs_phys

if "a_over_rs" in df.columns:
    a_over_rs_obs = pd.to_numeric(df["a_over_rs"], errors="coerce")
    df["a_over_rs_consistency"] = np.abs(a_over_rs_obs - a_over_rs_phys) / a_over_rs_phys

# Planet radius (Re) from ror and R*
ror = pd.to_numeric(df.get("ror"), errors="coerce")
df["Rp_Re_from_ror"] = ror * rad_Rs * (R_SUN / 6.371e6)  # R☉ to m, / R⊕

# Geometric transit probability P_tr ≈ (R*/a) * (1 + ror) / (1 - e^2)
ecc = pd.to_numeric(df.get("ecc"), errors="coerce")
df["geom_transit_prob"] = (rad_Rs*R_SUN / np.where(a_m>0, a_m, np.nan)) * (1.0 + ror.fillna(0.0)) / np.where(1 - (ecc.fillna(0.0)**2) > 0, 1 - (ecc.fillna(0.0)**2), np.nan)
df["geom_transit_prob"] = df["geom_transit_prob"].clip(upper=1.0)

# Equilibrium temperature (A=0.3, full redistribution) if missing
Teff = pd.to_numeric(df.get("teff_K"), errors="coerce")
A = 0.3
with np.errstate(invalid="ignore"):
    Teq_phys = Teff * np.sqrt( (rad_Rs*R_SUN) / (2.0 * a_m) ) * ((1 - A) ** 0.25)
df["Teq_phys_K"] = Teq_phys
# If eqt_K is present but NaN, fill
if "eqt_K" in df.columns:
    df["eqt_K"] = pd.to_numeric(df["eqt_K"], errors="coerce").fillna(Teq_phys)

# Transit duration consistency (small-angle analytic T14)

# T14 ≈ (P/pi) * (R*/a) * sqrt((1+ror)^2 - b^2) / sin(i)
dur_h = pd.to_numeric(df.get("dur_h"), errors="coerce")
impact = pd.to_numeric(df.get("impact"), errors="coerce")
incl = np.deg2rad(pd.to_numeric(df.get("incl_deg"), errors="coerce"))
b = impact
term = np.sqrt(np.maximum((1.0 + ror.fillna(0.0))**2 - np.maximum(b,0)**2, 0))
with np.errstate(invalid="ignore"):
    T14_model_h = (P_days/np.pi) * (rad_Rs*R_SUN/np.where(a_m>0,a_m,np.nan)) * term / np.sin(incl)
df["dur_model_h"] = T14_model_h
df["dur_consistency"] = np.abs(dur_h - T14_model_h) / np.where(dur_h>0, dur_h, np.nan)

# Baseline BLS S/N (classical benchmark)
try:
    # Requires time-series input (placeholder here for per-target LC injection in Step 6)
    # For Step 2, approximate with depth / scatter as proxy
    df["bls_sn_proxy"] = df["depth_ppm"] / (df["depth_ppm_err_mean"].replace(0, np.nan))
except Exception as e:
    print("⚠️ BLS proxy failed:", e)
    df["bls_sn_proxy"] = np.nan
    
# Reduced proper motion (use J/H/G if available) for dwarf/giant proxy
def reduced_PM(mag, pm_masyr):
    mu_arcsec_yr = (pd.to_numeric(pm_masyr, errors="coerce").abs()) / 1000.0
    return pd.to_numeric(mag, errors="coerce") + 5.0*np.log10(mu_arcsec_yr) + 5.0

df["H_G"] = reduced_PM(df.get("mag_T"), df.get("pm_tot"))  # proxy if Gaia G unavailable; TESS mag used
df["H_K"] = reduced_PM(df.get("mag_Kepler"), df.get("pm_tot"))

# Giant flag (very rough): low PM + bright mag + low logg
df["giant_flag"] = ((df.get("logg_cgs") < 3.8) | (pd.to_numeric(df.get("pm_tot"), errors="coerce") < 3.0)).astype(int)

# Photoevaporation valley indicator (simple boundary in logR–logS)
Rp = pd.to_numeric(df.get("Rp_Re_from_ror"), errors="coerce").fillna(pd.to_numeric(df.get("rade_Re"), errors="coerce"))
S = pd.to_numeric(df.get("insol_S"), errors="coerce")
logR = safe_log10(Rp)
logS = safe_log10(S)
# A simple sloped cut; planets near/below the valley often have different demographics
valley = (logR < (0.26 + 0.11*logS))  # heuristic boundary
df["valley_flag"] = valley.astype(int)

# Transit probability & duration sanity (flags)
df["low_prob_flag"] = (df["geom_transit_prob"] < 0.005).astype(int)
df["dur_outlier_flag"] = (df["dur_consistency"] > 0.5).astype(int)
# Physics veto: flag systems that break basic transit physics
df["physics_veto_flag"] = (
    ((df["dur_consistency"] > 1.0).fillna(False)) |
    ((df["geom_transit_prob"] < 1e-3).fillna(False))
).astype(int)
# ---------- end physics augmentation ----------

# Build feature list
feature_cols = gather_features(df)
X = df[feature_cols].copy()

# Impute numeric with median (per-feature) – simple & stable for trees
medians = {c: X[c].median() for c in feature_cols}
for c in feature_cols:
    X[c] = X[c].fillna(medians[c])

X = X.to_numpy().astype(float)

print(f"✅ Features: {len(feature_cols)} columns")
print("Some features:", feature_cols[:20], "...")

# =========================================================
# CV splitter (anti-leakage via star_id grouping)
# =========================================================
if HAS_SGKF:
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    cv_iter = splitter.split(X, y, groups=groups)
else:
    # Fallback if sklearn version lacks StratifiedGroupKFold:
    # stratify by y; group leakage reduced by grouping same stars in folds using a hash bucketing
    print("⚠️ StratifiedGroupKFold not available; using StratifiedKFold fallback (group leakage risk reduced but not eliminated).")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    cv_iter = skf.split(X, y)

# =========================================================
# Train base models with OOF predictions
# =========================================================
model_specs = []
if HAS_XGB: model_specs.append(("xgb", "XGBoost", PARAM_XGB))
if HAS_LGBM: model_specs.append(("lgbm", "LightGBM", PARAM_LGBM))
if HAS_CAT: model_specs.append(("cat", "CatBoost", PARAM_CAT))

if not model_specs:
    raise RuntimeError("No gradient boosting libraries found. Please install xgboost, lightgbm, or catboost.")

oof_proba = {name: np.zeros((len(y), N_CLASSES), dtype=float) for name,_,_ in model_specs}
fold_scores = defaultdict(list)
models_per_fold = defaultdict(list)

cls_weights = class_weight_vector(y)

for fold, (tr_idx, va_idx) in enumerate(cv_iter, start=1):
    print(f"\n🧪 Fold {fold}/{N_SPLITS}")
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]

    # Per-fold class weights
    cw = class_weight_vector(ytr)
    # Inverse uncertainty weights (depth_ppm error as proxy)
    unc = df["depth_ppm_err_mean"].to_numpy()
    inv_unc = 1.0 / np.where((unc > 0) & np.isfinite(unc), unc, 1.0)
    # Apply only to training set
    sw_unc = inv_unc[tr_idx]

    # Combine class weights and uncertainty
    sw = np.array([cw[c] for c in ytr]) * sw_unc


    for key, label, params in model_specs:
        print(f"  ▶ Training {label}...")
        t0 = time.time()  # ⏱ start timer
        if key == "xgb":
            model = XGBClassifier(**params)
            model.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], verbose=False)
            proba = model.predict_proba(Xva)
        elif key == "lgbm":
            model = LGBMClassifier(**params)
            model.fit(Xtr, ytr, sample_weight=sw)
            proba = model.predict_proba(Xva)
        elif key == "cat":
            model = CatBoostClassifier(**params)
            cw_vec = [cw.get(i, 1.0) for i in range(N_CLASSES)]
            model.set_params(class_weights=cw_vec)
            model.fit(Xtr, ytr, sample_weight=sw, eval_set=(Xva, yva), verbose=False)
            proba = model.predict_proba(Xva)
        else:
            continue
        elapsed = time.time() - t0
        fold_scores[(key,"time_sec")].append(elapsed)
        print(f"    Training time: {elapsed:.2f}s")

        oof_proba[key][va_idx] = proba
        # Save fold model for later refit pattern reference (we’ll refit on all data after CV)
        models_per_fold[key].append(model)

        pr_macro, per_class_pr = pr_auc_macro(yva, proba)
        roc_ovo = roc_auc_ovo(yva, proba)
        fold_scores[(key,"pr_auc_macro")].append(pr_macro)
        fold_scores[(key,"roc_auc_ovo")].append(roc_ovo)
        print(f"    PR-AUC (macro): {pr_macro:.4f} | ROC-AUC (ovo): {roc_ovo:.4f}")

# Base model OOF evaluation & weights for ensembling
base_report = {}
weights = {}
for key, label, _ in model_specs:
    pr_macro, per_class = pr_auc_macro(y, oof_proba[key])
    roc_ovo = roc_auc_ovo(y, oof_proba[key])
    base_report[key] = {
        "label": label,
        "oof_pr_auc_macro": pr_macro,
        "oof_pr_auc_per_class": per_class,
        "oof_roc_auc_ovo": roc_ovo,
        "cv_pr_auc_macro_mean": float(np.mean(fold_scores[(key,"pr_auc_macro")])),
        "cv_pr_auc_macro_std": float(np.std(fold_scores[(key,"pr_auc_macro")])),
    }
    # Weight by squared PR-AUC macro (emphasize strong models)
    weights[key] = pr_macro ** 2

# Normalize weights
w_sum = sum(weights.values())
if w_sum == 0:
    weights = {k: 1.0/len(weights) for k in weights}
else:
    weights = {k: v / w_sum for k, v in weights.items()}

print("\n🔗 Base-model OOF summary:")
for k,v in base_report.items():
    print(f"  {v['label']}: PR-AUC(macro)={v['oof_pr_auc_macro']:.4f}  ROC-AUC(ovo)={v['oof_roc_auc_ovo']:.4f}  weight={weights[k]:.3f}")

# =========================================================
# Speed vs Score chart
# =========================================================
times = [np.mean(fold_scores[(k,"time_sec")]) for k,_,_ in model_specs]
scores = [base_report[k]["oof_pr_auc_macro"] for k,_,_ in model_specs]
labels = [label for _,label,_ in model_specs]

plt.figure(figsize=(6,5))
plt.scatter(times, scores, s=80)
for t,s,l in zip(times, scores, labels):
    plt.text(t, s, l)
plt.xlabel("Avg Training Time per Fold (s)")
plt.ylabel("PR-AUC (macro)")
plt.title("Speed vs Score Tradeoff (per Electronics 2024)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_REPORTS/"speed_vs_score.png", dpi=160)
plt.close()

# =========================================================
# Calibrated stacking (meta logistic regression on OOF probs)
# =========================================================
oof_stack = np.zeros_like(next(iter(oof_proba.values())))
for k,proba in oof_proba.items():
    oof_stack += weights[k] * proba

# Meta-model: logistic regression on concatenated base proba (or on weighted average)
# Here we fit on concatenated base probs for richer signal
stack_matrix = np.hstack([oof_proba[k] for k,_,_ in model_specs])  # shape (n, 3 * n_models)

meta = LogisticRegression(
    C=2.0, max_iter=1000, class_weight="balanced", random_state=42
)
meta.fit(stack_matrix, y)

# Calibrate the stacked probs with isotonic (per-class via one-vs-rest calibration)
# (We’ll calibrate a simple logistic regression on the weighted-average probs for stability)
cal_meta = CalibratedClassifierCV(
    estimator=OneVsRestClassifier(LogisticRegression(max_iter=1000)),
    method="isotonic", cv=3
)
cal_meta.fit(oof_stack, y)

# OOF evaluation of stacked + calibrated
pr_macro_stack, per_class_stack = pr_auc_macro(y, oof_stack)
pr_macro_cal, per_class_cal = pr_auc_macro(y, ensure_numpy(cal_meta.predict_proba(oof_stack)))

print(f"\n🧮 OOF Ensemble (weighted) PR-AUC macro: {pr_macro_stack:.4f}")
print(f"🧮 OOF Ensemble (calibrated) PR-AUC macro: {pr_macro_cal:.4f}")

# Confusion matrix at argmax
y_pred_oof = np.argmax(oof_stack, axis=1)
plot_confusion(y, y_pred_oof, "OOF Confusion – Weighted Ensemble", OUT_REPORTS/"oof_confusion_weighted.png")
plot_pr_curves(y, oof_stack, "OOF Weighted Ensemble", str(OUT_REPORTS/"oof_weighted"))

y_pred_cal = np.argmax(cal_meta.predict_proba(oof_stack), axis=1)
plot_confusion(y, y_pred_cal, "OOF Confusion – Calibrated Ensemble", OUT_REPORTS/"oof_confusion_calibrated.png")
plot_pr_curves(y, cal_meta.predict_proba(oof_stack), "OOF Calibrated Ensemble", str(OUT_REPORTS/"oof_calibrated"))

# =========================================================
# Per-mission diagnostics
# =========================================================
mission_reports = {}
for mission, flag in [("Kepler","is_kepler"), ("K2","is_k2"), ("TESS","is_tess")]:
    if flag in df.columns and df[flag].sum() > 0:
        mask = df[flag] == 1
        y_m = y[mask.values]
        if len(y_m) > 0:
            p_m = oof_stack[mask.values]
            pr_m, pc_m = pr_auc_macro(y_m, p_m)
            mission_reports[mission] = {"rows": int(mask.sum()), "oof_pr_auc_macro": pr_m, "per_class": pc_m}

# =========================================================
# Refit base models on ALL data for deployment
# =========================================================
fitted_models = {}
for key, label, params in model_specs:
    print(f"\n🧩 Refit on all data: {label}")
    if key == "xgb":
        model = XGBClassifier(**params)
        sw = np.array([cls_weights[c] for c in y])
        model.fit(X, y, sample_weight=sw, verbose=False)
    elif key == "lgbm":
        model = LGBMClassifier(**params)
        sw = np.array([cls_weights[c] for c in y])
        model.fit(X, y, sample_weight=sw)
    elif key == "cat":
        model = CatBoostClassifier(**params)
        cw_vec = [cls_weights.get(i, 1.0) for i in range(N_CLASSES)]
        model.set_params(class_weights=cw_vec)
        model.fit(X, y, verbose=False)
    fitted_models[key] = model

# Build deploy-time stacker on base probs
proba_full = np.hstack([fitted_models[k].predict_proba(X) for k,_,_ in model_specs])
meta_deploy = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", random_state=42)
meta_deploy.fit(proba_full, y)

# Calibrate deploy-time weighted average
weighted_full = np.zeros((len(y), N_CLASSES))
for k in fitted_models:
    weighted_full += weights[k] * fitted_models[k].predict_proba(X)
cal_meta_deploy = CalibratedClassifierCV(
    estimator=OneVsRestClassifier(LogisticRegression(max_iter=1000)),
    method="isotonic", cv=3
)
cal_meta_deploy.fit(weighted_full, y)
# =========================================================
# Save artifacts
# =========================================================
artifacts = {
    "feature_cols": feature_cols,
    "medians": medians,
    "weights": weights,
    "target_map": TARGET_MAP,
    "params": {
        "xgb": PARAM_XGB if HAS_XGB else None,
        "lgbm": PARAM_LGBM if HAS_LGBM else None,
        "cat": PARAM_CAT if HAS_CAT else None
    },
    "cv": {
        "n_splits": N_SPLITS,
        "stratified_group_cv": HAS_SGKF
    },
    "base_report": base_report,
    "ensemble_oof": {
        "pr_auc_macro_weighted": pr_macro_stack,
        "pr_auc_macro_calibrated": pr_macro_cal
    },
    "mission_reports": mission_reports
}
# =========================================================
# Reproducibility metadata (Model Card)
# =========================================================
import platform, hashlib
import sklearn, xgboost, lightgbm, catboost

with open(INFILE, "rb") as f:
    data_hash = hashlib.md5(f.read()).hexdigest()

artifacts["reproducibility"] = {
    "random_seed": 42,
    "python_version": platform.python_version(),
    "sklearn_version": sklearn.__version__,
    "xgboost_version": getattr(xgboost, "__version__", None),
    "lightgbm_version": getattr(lightgbm, "__version__", None),
    "catboost_version": getattr(catboost, "__version__", None),
    "dataset_hash": data_hash
}
# right before saving artifacts
medians_json = {c: float(v) if pd.notna(v) else None for c, v in medians.items()}
weights_json = {k: float(v) for k, v in weights.items()}
artifacts["weights"] = weights_json
with open(OUT_MODELS/"features.json","w") as f:
    json.dump({"feature_cols": feature_cols,
               "medians": medians_json,
               "weights": weights_json,
               "target_map": TARGET_MAP}, f, indent=2)
with open(OUT_REPORTS/"training_report.json", "w") as f:
    json.dump(artifacts, f, indent=2)

# Save models
for k,m in fitted_models.items():
    joblib.dump(m, OUT_MODELS/f"{k}_model.pkl")
joblib.dump(meta_deploy, OUT_MODELS/"stack_meta.pkl")
joblib.dump(cal_meta_deploy, OUT_MODELS/"cal_meta.pkl")
with open(OUT_MODELS/"features.json","w") as f:
    json.dump({"feature_cols": feature_cols, "medians": medians, "weights": weights, "target_map": TARGET_MAP}, f, indent=2)

print("\n✅ Step 2 complete.")
print("Saved:")
print("  - reports/training_report.json")
print("  - reports/oof_confusion_weighted.png, reports/oof_confusion_calibrated.png")
print("  - reports/oof_weighted_pr_*.png, reports/oof_calibrated_pr_*.png")
print("  - models/*_model.pkl, models/stack_meta.pkl, models/cal_meta.pkl, models/features.json")
