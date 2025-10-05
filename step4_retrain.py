#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 4 — Dynamic Retraining Pipeline (NASA Edition)

Features:
- Ingest KOI/K2/TOI-like CSVs into unified schema
- Append to processed/exo_unified.parquet (safe w/ backups)
- Run Step 3 end-to-end (ensemble + calibration + physics-aware)
- Snapshot models/reports/logs under registry/v<timestamp>
- Compute deltas vs previous release
- Refresh SHAP + reliability plots
- Discovery Mode: score unlabeled pool, export high-confidence candidates
"""

import argparse, datetime as dt, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from step1 import compute_engineered_features  # <-- add this import at the top
from step5_explain_discovery import run_step5

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed"; PROCESSED.mkdir(exist_ok=True)
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)
REPORTS = ROOT / "reports"; REPORTS.mkdir(exist_ok=True)
LOGS = ROOT / "logs"; LOGS.mkdir(exist_ok=True)
REGISTRY = ROOT / "registry"; REGISTRY.mkdir(exist_ok=True)

UNIFIED_PARQUET = PROCESSED / "exo_unified.parquet"
UNLABELED_POOL  = PROCESSED / "exo_unlabeled_pool.parquet"
STEP3_SCRIPT    = ROOT / "step3_unified_model.py"

TARGET_MAP = {0:"False Positive",1:"Candidate",2:"Confirmed"}

# Leakage cols we never want in training
LEAKY_COLS = ["label_text","human_vetting_notes","is_confirmed_truth",
              "disposition","koi_score","tfopwg_disp"]

def _now_tag():
    return dt.datetime.utcnow().strftime("v%Y%m%d-%H%M%S-UTC")

def _save_feature_medians(df: pd.DataFrame):
    medians = {}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            val = df[c].median(skipna=True)
            # Ensure float and safe fallback
            if pd.notna(val):
                medians[c] = float(val)
            else:
                medians[c] = float(np.nanmedian(df[c].to_numpy()))
    (PROCESSED / "exo_unified.meta.json").write_text(
        json.dumps({"medians": medians}, indent=2)
    )
    print("📐 Medians updated → exo_unified.meta.json")

def _previous_release_dir(current_release: str):
    # returns Path or None
    dirs = sorted(
        [d for d in REGISTRY.iterdir() if d.is_dir() and d.name.startswith("v")],
        key=lambda p: p.name
    )
    if len(dirs) < 2:
        return None
    # last is current, second last is previous
    if dirs[-1].name == current_release:
        return dirs[-2]
    # fallback: pick second last anyway
    return dirs[-2]


# ----------------------------
# Ingest + Schema
# ----------------------------
def _safe_numeric(s): return pd.to_numeric(s, errors="coerce")

def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower():c for c in df.columns}
    mappings = {}

    # Example mission mappings
    if "koi_period" in lower: mappings[lower["koi_period"]] = "period"
    if "toi_period" in lower: mappings[lower["toi_period"]] = "period"
    if "kepmag" in lower: mappings[lower["kepmag"]] = "mag_Kepler"
    if "tmag" in lower: mappings[lower["tmag"]] = "mag_T"
    df = df.rename(columns=mappings)

    # Flags
    is_kepler = int("koi_" in " ".join(lower))
    is_tess   = int("toi_" in " ".join(lower) or "tmag" in lower)
    is_k2     = int("k2" in " ".join(lower))
    df["is_kepler"] = df.get("is_kepler", pd.Series([is_kepler]*len(df)))
    df["is_k2"]     = df.get("is_k2", pd.Series([is_k2]*len(df)))
    df["is_tess"]   = df.get("is_tess", pd.Series([is_tess]*len(df)))

    # star_id
    if "star_id" not in df.columns:
        for cand in ["kepid","epic","tic","id","star"]:
            if cand in lower:
                df["star_id"] = df[lower[cand]].astype(str)
                break
    if "star_id" not in df.columns:
        df["star_id"] = "anon_" + pd.util.hash_pandas_object(df.index, index=True).astype(str)

    # Drop leaks
    df = df.drop(columns=[c for c in LEAKY_COLS if c in df.columns], errors="ignore")

    # label_int normalize
    if "label_int" in df:
        mapping = {"fp":0,"false positive":0,"candidate":1,"cand":1,"confirmed":2,"conf":2}
        if df["label_int"].dtype == object:
            df["label_int"] = df["label_int"].str.strip().str.lower().map(mapping)
        df["label_int"] = pd.to_numeric(df["label_int"], errors="coerce")
    # Ensure label_int column exists (unlabeled ingestion otherwise KeyErrors later)
    if "label_int" not in df.columns:
        df["label_int"] = np.nan
    return df

def ingest_csv(path: Path):
    raw = pd.read_csv(path, low_memory=False)
    df = _ensure_schema(raw)
    # 🚀 Add engineered physics-aware features (Step 1 functions)
    df = compute_engineered_features(df)

    # 🔭 Physics integrity check (Malik 2022)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)

    labeled = df[df["label_int"].notna()].copy()
    unlabeled = df[df["label_int"].isna()].copy()

    if UNIFIED_PARQUET.exists():
        backup = PROCESSED / f"exo_unified.{_now_tag()}.bak.parquet"
        shutil.copy2(UNIFIED_PARQUET, backup)
        prev = pd.read_parquet(UNIFIED_PARQUET)
        combined = pd.concat([prev,labeled],ignore_index=True)
    else:
        combined = labeled
    # 🧹 De-duplicate to avoid double-counting across incremental ingests
    dedup_keys = [k for k in ["star_id", "period"] if k in combined.columns]
    if dedup_keys:
        combined = combined.drop_duplicates(subset=dedup_keys, keep="last").reset_index(drop=True)
    combined.to_parquet(UNIFIED_PARQUET, index=False)
    
    # 🚀 Recompute medians for imputation
    _save_feature_medians(combined)

    if len(unlabeled):
        pool = pd.read_parquet(UNLABELED_POOL) if UNLABELED_POOL.exists() else pd.DataFrame()
        # Align to union of columns so schema stays stable across releases
        all_cols = list(set(pool.columns).union(unlabeled.columns))
        pool = pool.reindex(columns=all_cols)
        unlabeled = unlabeled.reindex(columns=all_cols)
        pd.concat([pool, unlabeled], ignore_index=True).to_parquet(UNLABELED_POOL, index=False)

# ----------------------------
# Training + Snapshot
# ----------------------------
def run_step3_and_snapshot(notes=""):
    release = _now_tag()
    rel_dir = REGISTRY / release
    rel_models = rel_dir/"models"; rel_reports = rel_dir/"reports"; rel_logs = rel_dir/"logs"
    for d in [rel_models,rel_reports,rel_logs]: d.mkdir(parents=True,exist_ok=True)

    log_path = rel_logs/"train.log"
    with open(log_path,"w") as f:
        ret = subprocess.call([sys.executable,str(STEP3_SCRIPT)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if ret!=0: raise RuntimeError("Step 3 failed; see logs")

    for p in REPORTS.glob("*"): shutil.copy2(p,rel_reports/p.name)
    for p in MODELS.glob("*"): shutil.copy2(p,rel_models/p.name)

    rep = REPORTS/"step3_report.json"
    data = json.loads(rep.read_text()) if rep.exists() else {}
    latest = REGISTRY/"latest"
    if latest.exists(): latest.unlink()
    try: os.symlink(rel_dir,latest,target_is_directory=True)
    except: (REGISTRY/"LATEST.txt").write_text(str(rel_dir))

    summary = {
        "release": release,
        "notes": notes,
        "scores": {
            "oof": data.get("scores", {}).get("oof", {}),
            "holdout": data.get("scores", {}).get("holdout", {}),
            "ece": data.get("scores", {}).get("ece", None),
            "per_mission": data.get("scores", {}).get("per_mission", {})
        },
        "thresholds": data.get("best_thresholds", {}),
        "conformal_q": data.get("conformal_q", None),
        "paths": {"models": str(rel_models), "reports": str(rel_reports)}
    }
    (rel_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary

# ----------------------------
# Delta Score Comparison (Fix 4)
# ----------------------------
def _delta_scores(curr, prev):
    if not prev:
        return {}

    def _diff(a, b):
        try:
            return float(a) - float(b)
        except Exception:
            return None

    return {
        # Out-of-fold deltas
        "oof_pr_auc_macro_delta": _diff(
            curr["scores"].get("oof", {}).get("calibrated_pr_auc_macro"),
            prev["scores"].get("oof", {}).get("calibrated_pr_auc_macro")
        ),
        # Holdout deltas
        "holdout_pr_auc_macro_delta": _diff(
            curr["scores"].get("pr_auc_macro"),
            prev["scores"].get("pr_auc_macro")
        ),
        "holdout_roc_auc_ovo_delta": _diff(
            curr["scores"].get("roc_auc_ovo"),
            prev["scores"].get("roc_auc_ovo")
        ),
        # Calibration / trust
        "ece_delta": _diff(
            curr["scores"].get("ece"),
            prev["scores"].get("ece")
        ),
        # Per-mission PR-AUC shifts
        "per_mission_delta": {
            m: _diff(
                curr["scores"].get("per_mission", {}).get(m),
                prev["scores"].get("per_mission", {}).get(m)
            )
            for m in ["kepler", "k2", "tess"]
        }
    }

# ----------------------------
# SHAP Shift Tracking (Fix 5)
# ----------------------------
def _compare_shap(curr, prev):
    try:
        shap_c = curr.get("shap_values", {})
        shap_p = prev.get("shap_values", {})
        return {f: shap_c[f] - shap_p.get(f, 0) for f in shap_c}
    except Exception:
        return {}

def run_discovery_candidates(release):
    if not UNLABELED_POOL.exists(): 
        return None
    df = pd.read_parquet(UNLABELED_POOL)
    from step3_unified_model import predict_step3
    out = predict_step3(df.copy())
    P, preds = out["proba_calibrated"], out["pred_costaware"]

    df_out = df.copy()
    df_out["pred"] = preds
    df_out["p_conf"] = P[:,2]
    df_out["p_cand"] = P[:,1]
    df_out["p_fp"] = P[:,0]

    # 🔭 NEW: ensure mission flags exist
    for flag in ["is_kepler", "is_k2", "is_tess"]:
        if flag not in df_out.columns:
            df_out[flag] = pd.NA

    sel = (P[:,2] >= 0.9) | (P[:,1] >= 0.95)
    out_path = REGISTRY / release / "reports" / "discovery_candidates.csv"
    df_out[sel].to_csv(out_path, index=False)
    return str(out_path)


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_csv",type=str)
    ap.add_argument("--notes",type=str,default="")
    ap.add_argument("--no-train",action="store_true")
    ap.add_argument("--discovery",action="store_true")
    args = ap.parse_args()

    if args.new_csv:
        summ = ingest_csv(Path(args.new_csv))
        print("📊 Ingest:",summ)

    if args.no_train: sys.exit(0)

    current = run_step3_and_snapshot(notes=args.notes)
    print("✅ Snapshot:",current)

    # 🚀 NEW: deltas + shap shift
    prev_dir = _previous_release_dir(current["release"])
    prev = None
    if prev_dir and (prev_dir / "summary.json").exists():
        try:
            prev = json.loads((prev_dir / "summary.json").read_text())
        except Exception:
            prev = None

    deltas = _delta_scores(current, prev)
    shap_shift = _compare_shap(current, prev)

    final = {
        "ingest": summ if args.new_csv else None,   # you could hook in ingest summary here
        "current_release": current,
        "previous_release": prev.get("release") if prev else None,
        "deltas_vs_previous": deltas,
        "shap_shift": shap_shift
    }

    print("\n🧾 FINAL SUMMARY")
    print(json.dumps(final, indent=2))
    (REGISTRY / "SCOREBOARD.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )
    # 🚀 Auto-run Step 5 after retraining
    try:
        step5_out = run_step5(out_prefix=f"{current['release']}_step5")
        print("🔭 Step 5 Explainability + Candidates generated:", step5_out)
    except Exception as e:
        print(f"[WARN] Step 5 failed: {e}")

    if args.discovery:
        out = run_discovery_candidates(current["release"])
        if out: print(f"🪄 Discovery candidates → {out}")

if __name__=="__main__":
    main()
