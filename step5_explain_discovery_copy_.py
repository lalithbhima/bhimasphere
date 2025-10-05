#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 5 — Explainability + Candidate Discovery (NASA Edition)

Outputs:
- reports/step5_explanations.jsonl         (per-row local explanations)
- reports/step5_candidates.csv             (ranked candidate list)
- reports/step5_summary.json               (run summary + knobs used)
- reports/step5_feature_insights.json      (global insights if available)
- registry/<release>/reports/...           (also copied by Step 4 snapshot)

Depends on Step 3 artifacts + Step 4 datasets.
"""

import json, os
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import sys

# Reuse Step 3 helpers
from step3_unified_model import (
    load_step3_bundle,
    physics_veto_adjust,
    N_CLASSES,
    TARGET_MAP
)

# Project paths
ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed"
REPORTS   = ROOT / "reports"
MODELS    = ROOT / "models"
REPORTS.mkdir(exist_ok=True)

UNLABELED_POOL = PROCESSED / "exo_unlabeled_pool.parquet"
UNIFIED_PARQUET = PROCESSED / "exo_unified.parquet"

# ---------- Config knobs ----------
CAND_THRESH_CONF = 0.90   # Confirmed prob threshold
CAND_THRESH_CAND = 0.95   # Candidate prob threshold
MAX_ROWS_EXPLAIN = 50000  # safety cap for batch explain
TOP_FEATURES_PER_SAMPLE = 8
RANKING_WEIGHTS = {
    # Transparent prioritization for follow-up
    "p_conf":  1.00,
    "p_cand":  0.40,
    "p_fp":   -0.60,
    "depth_snr": 0.15,
    "geom_transit_prob": 0.10,
    "dur_consistency_penalty": -0.10,  # (1 + |consistency|)^-1 is applied
}

# ---------- Utility ----------
def _safe_series(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series([default]*len(df))

def _threshold_margins(P_row, thresholds):
    # “how far above each class threshold we are”
    # thresholds: dict[int->float]
    margins = []
    for c in range(P_row.shape[0]):
        t = thresholds.get(str(c), thresholds.get(c, 0.5))
        margins.append(float(P_row[c] - float(t)))
    return margins

def _human_rationale(row, feat_names, shap_vals, margins, pred_set, veto_hit, pconf_raw=None, pconf_veto=None):
    parts = []
    if veto_hit:
        if pconf_raw is not None and pconf_veto is not None:
            parts.append(f"Physics veto reduced Confirmed P from {pconf_raw:.3f} → {pconf_veto:.3f} (b > 1+ror or ror ≥ 1).")
        else:
            parts.append("Physics veto applied (b > 1 + ror or ror ≥ 1 damped Confirmed).")
    if pred_set and len(pred_set) > 1:
        parts.append(f"Conformal set includes {', '.join(TARGET_MAP[c] for c in pred_set)} (uncertainty-aware).")
    # Feature rationale (top positive drivers toward the chosen class)
    if isinstance(shap_vals, np.ndarray):
        # pick largest magnitude features
        idx = np.argsort(-np.abs(shap_vals))[:min(5, len(shap_vals))]
        ranked = [f"{feat_names[i]} ({'+' if shap_vals[i]>=0 else '−'}{abs(shap_vals[i]):.2f})" for i in idx]
        parts.append("Top drivers: " + ", ".join(ranked))
    # Threshold margins summary
    lead = np.argmax(margins)
    parts.append(f"Best threshold margin: {TARGET_MAP[lead]} (+{margins[lead]:.3f}).")
    return " ".join(parts)

def _discovery_score(row):
    # Linear combination w/ simple transforms to keep it transparent
    p_conf = float(row.get("p_conf", 0))
    p_cand = float(row.get("p_cand", 0))
    p_fp   = float(row.get("p_fp", 0))
    depth_snr = float(row.get("depth_snr", 0))
    gtp   = float(row.get("geom_transit_prob", 0))
    dcon  = float(row.get("dur_consistency", np.nan))
    dcon_pen = 1.0 / (1.0 + abs(dcon)) if np.isfinite(dcon) else 1.0

    score = (
        RANKING_WEIGHTS["p_conf"] * p_conf +
        RANKING_WEIGHTS["p_cand"] * p_cand +
        RANKING_WEIGHTS["p_fp"]   * p_fp   +
        RANKING_WEIGHTS["depth_snr"] * depth_snr +
        RANKING_WEIGHTS["geom_transit_prob"] * gtp +
        RANKING_WEIGHTS["dur_consistency_penalty"] * (1.0 - dcon_pen)
    )
    return float(score)

def _load_local_shap(fitted_models, feature_cols):
    """
    Prefer LightGBM SHAP for local explanations; fallback to XGB/CAT if needed.
    Returns: (backend, explainer) or (None, None)
    """
    try:
        import shap
    except Exception:
        return None, None

    if "lgbm" in fitted_models:
        try:
            expl = shap.TreeExplainer(fitted_models["lgbm"])
            return ("lgbm", expl)
        except Exception:
            pass
    if "xgb" in fitted_models:
        try:
            expl = shap.TreeExplainer(fitted_models["xgb"])
            return ("xgb", expl)
        except Exception:
            pass
    if "cat" in fitted_models:
        try:
            expl = shap.TreeExplainer(fitted_models["cat"])
            return ("cat", expl)
        except Exception:
            pass
    return None, None

# ---------- Main API ----------
def run_step5(max_rows=None, out_prefix="step5"):
    """
    Generates explanations for all (or a capped subset of) rows in the unlabeled pool,
    ranks candidates, and writes artifacts for the UI and registry.
    """
    if not UNLABELED_POOL.exists():
        print("No unlabeled pool found; skipping Step 5.")
        return None

    feats, models, meta, cal, dr, q = load_step3_bundle()
    feature_cols = feats["feature_cols"]
    medians = feats["medians"]
    weights = feats["weights"]
    thresholds = json.load(open(ROOT / "reports" / "step3_report.json")).get("best_thresholds", {})

    # Prepare data
    df_raw = pd.read_parquet(UNLABELED_POOL)
    if max_rows is None:
        max_rows = MAX_ROWS_EXPLAIN
    if len(df_raw) > max_rows:
        print(f"[Step5] Capping from {len(df_raw)} to {max_rows} rows for explainability.")
        df_raw = df_raw.head(max_rows)

    Xdf = df_raw.reindex(columns=feature_cols)
    for c in feature_cols:
        Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce").fillna(medians.get(c, 0.0))
    X = Xdf.to_numpy().astype(float)

    # Base ensemble weighted probs
    P_weighted = np.zeros((len(X), N_CLASSES))
    for k, mdl in models.items():
        Pk = mdl.predict_proba(X)
        P_weighted += weights[k] * Pk
    
    # Save raw probabilities before physics veto
    P_raw = P_weighted.copy()

    # Physics veto + calibration
    P_weighted = physics_veto_adjust(P_weighted, Xdf)
    P_cal = cal.predict_proba(P_weighted)

    # Conformal sets
    def _in_conformal_set(row_probs, c, qdict):
        qc = qdict.get(str(c), qdict.get(c, None))
        if qc is None:
            return False
        return (1.0 - row_probs[c]) <= float(qc)

    conformal_sets = []
    for i in range(len(X)):
        row = P_cal[i]
        S = [c for c in range(N_CLASSES) if _in_conformal_set(row, c, q)]
        conformal_sets.append(S)

    # Local SHAP (if available)
    backend, shap_explainer = _load_local_shap(models, feature_cols)
    SHAP_OK = shap_explainer is not None

    # Build explanations JSONL + candidates CSV
    expl_path = REPORTS / f"{out_prefix}_explanations.jsonl"
    cand_path = REPORTS / f"{out_prefix}_candidates.csv"
    summary_path = REPORTS / f"{out_prefix}_summary.json"
    insights_path = REPORTS / f"{out_prefix}_feature_insights.json"

    out_rows = []
    with open(expl_path, "w") as f:
        for i in range(len(X)):
            prow = P_cal[i]
            pred = int(np.argmax(prow))
            margins = _threshold_margins(prow, thresholds)
            veto_hit = bool(P_raw[i,2] > P_weighted[i,2] + 1e-9)
            pconf_raw = float(P_raw[i, 2])
            pconf_veto = float(P_weighted[i, 2])

            # SHAP for the predicted class if available
            shap_vec = None
            if SHAP_OK:
                try:
                    sv = shap_explainer.shap_values(X[i:i+1], check_additivity=False)
                    # normalize output: list (n_classes) → take predicted class
                    if isinstance(sv, list):
                        shap_vec = np.asarray(sv[pred]).reshape(-1)
                    else:
                        shap_vec = np.asarray(sv).reshape(-1)  # some backends
                except Exception:
                    shap_vec = None

            rationale = _human_rationale(
                row=df_raw.iloc[i],
                feat_names=feature_cols,
                shap_vals=shap_vec if shap_vec is not None else np.zeros(len(feature_cols)),
                margins=margins,
                pred_set=conformal_sets[i],
                veto_hit=veto_hit,
                pconf_raw=pconf_raw,
                pconf_veto=pconf_veto,
            )

            rec = {
                "row_index": int(i),
                "star_id": str(df_raw.iloc[i].get("star_id", f"idx_{i}")),
                "pred": int(pred),
                "pred_text": TARGET_MAP[pred],
                "proba": [float(p) for p in prow],
                "threshold_margins": [float(m) for m in margins],
                "prediction_set": [int(c) for c in conformal_sets[i]],
                "rationale": rationale,
                "p_conf_before_veto": pconf_raw,
                "p_conf_after_veto":  pconf_veto
            }
            # attach top SHAPs
            if shap_vec is not None:
                idx = np.argsort(-np.abs(shap_vec))[:TOP_FEATURES_PER_SAMPLE]
                rec["top_features"] = [
                    {"feature": feature_cols[j], "shap": float(shap_vec[j])} for j in idx
                ]
            f.write(json.dumps(rec) + "\n")
            out_rows.append(rec)

    # Build candidate table
    df_out = pd.DataFrame(out_rows)
    df_out["p_fp"]   = df_out["proba"].apply(lambda v: float(v[0]))
    df_out["p_cand"] = df_out["proba"].apply(lambda v: float(v[1]))
    df_out["p_conf"] = df_out["proba"].apply(lambda v: float(v[2]))
    df_out["passes_gate"] = (df_out["p_conf"] >= CAND_THRESH_CONF) | (df_out["p_cand"] >= CAND_THRESH_CAND)

    # Attach science features used for ranking
    for aux in ["depth_snr", "geom_transit_prob", "dur_consistency"]:
        df_out[aux] = _safe_series(Xdf, aux, default=np.nan)
    # Attach science features used for ranking
    for aux in ["depth_snr", "geom_transit_prob", "dur_consistency"]:
        df_out[aux] = _safe_series(Xdf, aux, default=np.nan)

    # 🔭 NEW: Add mission flags & key physics columns for traceability
    for flag in ["is_kepler", "is_k2", "is_tess"]:
        if flag in df_raw.columns:
            df_out[flag] = df_raw[flag].astype("Int64")

    for phys in ["ror", "impact", "incl_deg", "period", "dur_h", "depth_ppm"]:
        if phys in Xdf.columns:
            df_out[phys] = pd.to_numeric(Xdf[phys], errors="coerce")

    df_out["discovery_score"] = df_out.apply(_discovery_score, axis=1)
    
    # Rank: gate first, then score
    df_out = df_out.sort_values(
        by=["passes_gate", "discovery_score", "p_conf", "p_cand"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    # Save CSV for judges & UI
    keep_cols = [
        "star_id","pred_text","p_conf","p_cand","p_fp",
        "discovery_score","passes_gate","threshold_margins","prediction_set",
        "depth_snr","geom_transit_prob","dur_consistency",
        "ror","impact","incl_deg","period","dur_h","depth_ppm",
        "is_kepler","is_k2","is_tess",
        "rationale"
    ]
    (df_out[keep_cols]).to_csv(cand_path, index=False)

    # Summary + knobs
    summary = {
        "rows_evaluated": int(len(X)),
        "gate_thresholds": {"Confirmed": CAND_THRESH_CONF, "Candidate": CAND_THRESH_CAND},
        "ranking_weights": RANKING_WEIGHTS,
        "explanations_path": str(expl_path),
        "candidates_path": str(cand_path),
        "shap_backend": backend if SHAP_OK else None
    }
    summary.update({
        "best_thresholds": thresholds,
        "weights_used": {k: float(v) for k, v in RANKING_WEIGHTS.items()}
    })
    (summary_path).write_text(json.dumps(summary, indent=2))

    # Optionally write global insights (if SHAP backend available)
    try:
        if SHAP_OK and "lgbm" in models:
            # global gain/importance from LightGBM as backup to SHAP summary plots from Step 3
            import numpy as np
            booster = models["lgbm"]
            gain = booster.booster_.feature_importance(importance_type="gain")
            split = booster.booster_.feature_importance(importance_type="split")
            g = np.asarray(gain).astype(float); s = np.asarray(split).astype(float)
            insights = [{"feature": feature_cols[i], "gain": float(g[i]), "split": float(s[i])} for i in range(len(feature_cols))]
            (insights_path).write_text(json.dumps({"global_importance": insights[:100]}, indent=2))
    except Exception:
        pass

    print(f"🧠 Explanations → {expl_path}")
    print(f"🪄 Candidates   → {cand_path}")
    print(f"🧾 Summary      → {summary_path}")
    return {
        "explanations": str(expl_path),
        "candidates": str(cand_path),
        "summary": str(summary_path)
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--out_prefix", type=str, default="step5")
    ap.add_argument("--pool", type=str, default=str(UNLABELED_POOL))
    args = ap.parse_args()

    # allow overriding pool path
    UNLABELED_POOL = Path(args.pool)
    # Fail-fast if Step 3 artifacts are missing
    try:
        feats, models, meta, cal, dr, q = load_step3_bundle()
    except Exception as e:
        print(f"[ERROR] Step 3 bundle not found or invalid: {e}")
        sys.exit(1)   # exit cleanly instead of return

    run_step5(max_rows=args.max_rows, out_prefix=args.out_prefix)
        
