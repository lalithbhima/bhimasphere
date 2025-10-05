# webapi/main.py
import os, io, json, math, time, shutil, hashlib, threading
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import subprocess, shutil, os, datetime


# ---- Project roots & artifacts (match your Steps 1–5) ----
ROOT = Path(__file__).resolve().parent   # not parent.parent
REPORTS = ROOT / "reports"
MODELS  = ROOT / "models"
PROCESSED = ROOT / "processed"
REGISTRY  = ROOT / "registry"

# Import Step-3 helpers (your code)
from step3_unified_model import load_step3_bundle, physics_veto_adjust  # 
# Step-4 helpers (ingest & retrain)
import step4_retrain as retrain_mod  # uses ingest, schema, snapshot, deltas  

# ------------------------- FastAPI -------------------------
app = FastAPI(title="NASA • A World Away • Step 6 API",
              version="1.0.0",
              description="Physics-aware, calibrated, conformal exoplanet classifier API (Steps 1–5 served)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Expose static assets (reports & registry) for PR curves, reliability, SHAP, CSVs, logs
if REPORTS.exists():
    app.mount("/reports", StaticFiles(directory=str(REPORTS), html=False), name="reports")
if REGISTRY.exists():
    app.mount("/registry", StaticFiles(directory=str(REGISTRY), html=False), name="registry")

# ---------------------- Bundle loader ----------------------
class Bundle:
    feats: Dict[str, Any]
    models: Dict[str, Any]
    meta: Dict[str, Any]
    calibrator: Any
    domain_ratio: Any
    conformal_q: Dict[str, float]
    thresholds: Dict[str, float]
    report: Dict[str, Any]

BUNDLE: Optional[Bundle] = None

def _load_report():
    rep = REPORTS / "step3_report.json"
    return json.loads(rep.read_text()) if rep.exists() else {}

def _load_thresholds_from_report(rep: dict) -> Dict[str, float]:
    # your Step-3 stores best thresholds here
    return rep.get("best_thresholds", {}) or {}

def _ensure_bundle(reload: bool=False):
    global BUNDLE
    if (BUNDLE is None) or reload:
        feats, models, meta, cal, dr, q = load_step3_bundle()  # loads from ./models  
        rep = _load_report()
        bundle = Bundle()
        bundle.feats = feats               # {"feature_cols","medians","weights","target_map"}
        bundle.models = models
        bundle.meta  = meta
        bundle.calibrator = cal
        bundle.domain_ratio = dr
        bundle.conformal_q = q or {}
        bundle.report = rep
        bundle.thresholds = _load_thresholds_from_report(rep)
        BUNDLE = bundle
    return BUNDLE

# -------------------- Data contracts -----------------------
TARGET_MAP = {0:"False Positive",1:"Candidate",2:"Confirmed"}  # aligns with Steps 2/3 

class PredictPayload(BaseModel):
    # Novice-friendly, all optional; server imputes using Step-3 medians
    # Transit / geometry
    period: Optional[float] = None
    dur_h: Optional[float] = Field(None, alias="duration_hours")
    depth_ppm: Optional[float] = None
    ror: Optional[float] = Field(None, alias="radius_ratio_rp_over_rs")
    a_over_rs: Optional[float] = None
    impact: Optional[float] = Field(None, alias="b_impact")
    ecc: Optional[float] = None
    incl_deg: Optional[float] = None
    # Stellar
    teff_K: Optional[float] = None
    logg_cgs: Optional[float] = None
    feh_dex: Optional[float] = None
    rad_Rs: Optional[float] = None
    mass_Ms: Optional[float] = None
    age_Gyr: Optional[float] = None
    # Astro/mags
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None
    pm_ra_masyr: Optional[float] = None
    pm_dec_masyr: Optional[float] = None
    dist_pc: Optional[float] = None
    mag_T: Optional[float] = None
    mag_Kepler: Optional[float] = None
    # Engineered (optional client preview; server recomputes authoritative set in Steps 1/3)
    depth_from_ror_ppm: Optional[float] = None
    geom_transit_prob: Optional[float] = None
    dur_model_h: Optional[float] = None
    dur_consistency: Optional[float] = None
    # Mission flag (optional)
    mission: Optional[str] = Field(None, description="Kepler|K2|TESS")

class PredictResult(BaseModel):
    proba: List[float]
    pred: int
    pred_text: str
    threshold_margins: List[float]
    conformal_set: List[int]
    physics_veto_applied: bool
    veto_reason: Optional[str]
    rationale: str
    top_features: List[Dict[str, Any]] = []

# -------------------- Utility: prediction -------------------
def _row_to_Xdf(row: Dict[str, Any], feature_cols: List[str], medians: Dict[str, Any]) -> pd.DataFrame:
    # map incoming json to full feature frame expected by Step-3
    Xdf = pd.DataFrame([{c: row.get(c, None) for c in feature_cols}])
    for c in feature_cols:
        Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce").fillna(medians.get(c, 0.0))
    return Xdf

def _weighted_proba(models: Dict[str, Any], weights: Dict[str, float], X: np.ndarray) -> np.ndarray:
    P = np.zeros((len(X), 3))
    for k, mdl in models.items():
        w = float(weights.get(k, 0.0))
        if w <= 0: continue
        P += w * mdl.predict_proba(X)
    return P

def _margins(P_row: np.ndarray, thresholds: Dict[str, Any]) -> List[float]:
    out = []
    for c in range(3):
        t = thresholds.get(str(c), thresholds.get(c, 0.5))
        out.append(float(P_row[c] - float(t)))
    return out

def _conformal_set(P_row: np.ndarray, q: Dict[str, float]) -> List[int]:
    S = []
    for c in range(3):
        qc = q.get(str(c), q.get(c, None))
        if qc is None: continue
        if (1.0 - P_row[c]) <= float(qc):  # same rule used in Step-5 
            S.append(c)
    return S

# --------------------------- Routes -------------------------

@app.get("/api/release")
def get_release():
    """Registry latest + Step-3 report, for reproducibility banner."""
    latest = (REGISTRY / "latest" / "summary.json")
    if latest.exists():
        data = json.loads(latest.read_text())
    else:
        data = {"release": None, "paths": {"models": str(MODELS), "reports": str(REPORTS)}}
    # augment with Step-3 report reproducibility block if present (Step-2/3 artifacts) 
    rep = _load_report()
    data["reproducibility"] = rep.get("reproducibility", data.get("reproducibility"))
    return data

@app.get("/api/metrics")
def get_metrics():
    """OOF/holdout PR-AUC/ROC-AUC, ECE, per-mission, thresholds, conformal q."""
    rep = _load_report()
    bundle = _ensure_bundle()
    return {
        "scores": rep.get("scores", {}),
        "best_thresholds": bundle.thresholds,
        "conformal_q": bundle.conformal_q,
        "weights": bundle.feats.get("weights", {}),
        "feature_count": len(bundle.feats.get("feature_cols", [])),
        "reproducibility": rep.get("reproducibility", {}),
    }

@app.get("/api/candidates")
def get_candidates(topK: Optional[int]=Query(None, ge=1),
                   mission: Optional[str]=Query(None),
                   gate: Optional[bool]=Query(None),
                   q: Optional[str]=Query(None),
                   page:int=Query(1, ge=1), size:int=Query(100, ge=1, le=1000)):
    """
    Paginated candidates from Step-5. Columns documented in your outline.
    """
    # Prefer registry/latest (from Step-4) else reports/step5_candidates.csv  
    cand = REGISTRY / "latest" / "reports" / "step5_candidates.csv"
    if not cand.exists():
        cand = REPORTS / "step5_candidates.csv"
    if not cand.exists():
        raise HTTPException(404, "No candidates CSV found. Generate via Step-5 or retrain.")
    df = pd.read_csv(cand)
    if mission:
        mm = mission.strip().lower()
        if mm in ("kepler","k2","tess"):
            col = f"is_{mm}"
            if col in df.columns:
                df = df[df[col].fillna(0).astype(int)==1]
    if gate is not None and "passes_gate" in df.columns:
        df = df[df["passes_gate"].astype(bool) == bool(gate)]
    if q:
        mask = df.apply(lambda r: q.lower() in str(r.get("star_id","")).lower() or
                                  q.lower() in str(r.get("rationale","")).lower(), axis=1)
        df = df[mask]
    if topK:
        df = df.head(topK)
    # pagination
    total = len(df)
    start = (page-1)*size
    end   = start+size
    rows = df.iloc[start:end].to_dict(orient="records")
    return {"total": int(total), "page": page, "size": size, "rows": rows}

@app.get("/api/explanations")
def get_explanations(star_id: Optional[str]=None, row_index: Optional[int]=None, topK:int=50):
    """Local explanations from Step-5 JSONL."""
    # Prefer registry/latest JSONL if present
    jl = REGISTRY / "latest" / "reports" / "step5_explanations.jsonl"
    if not jl.exists():
        jl = REPORTS / "step5_explanations.jsonl"
    if not jl.exists():
        raise HTTPException(404, "No explanations JSONL found.")
    out = []
    with open(jl, "r") as f:
        for i, ln in enumerate(f):
            try:
                rec = json.loads(ln)
                if star_id and str(rec.get("star_id")) != str(star_id): 
                    continue
                if (row_index is not None) and (int(rec.get("row_index",-1)) != int(row_index)):
                    continue
                out.append(rec)
                if len(out) >= topK: break
            except Exception:
                continue
    return {"count": len(out), "items": out}

@app.post("/api/predict", response_model=PredictResult)
def predict_one(payload: PredictPayload):
    """
    Single-row inference:
      - reindex to feature_cols
      - impute with medians (Step-3/Step-4 medians)
      - ensemble → physics veto → calibration → conformal set
      - threshold margins
    """
    bundle = _ensure_bundle()
    feats = bundle.feats
    feature_cols = feats["feature_cols"]
    medians = feats["medians"]
    weights = feats["weights"]

    # Map request to full feature vector
    row = payload.dict(by_alias=True, exclude_none=True)
    Xdf = _row_to_Xdf(row, feature_cols, medians)
    X = Xdf.to_numpy().astype(float)

    # Weighted base ensemble (pre-calibration)
    Pw = _weighted_proba(bundle.models, weights, X)

    # Save raw Confirmed prob for veto message
    pconf_raw = float(Pw[0,2])

    # Physics veto (uses Xdf columns) then isotonic calibration (Step-3)  
    P_phys = physics_veto_adjust(Pw, Xdf)
    P_cal  = bundle.calibrator.predict_proba(P_phys)
    P = P_cal[0]

    # Conformal set
    S = _conformal_set(P, bundle.conformal_q)

    # Threshold margins (best thresholds from Step-3 report)  
    margins = _margins(P, bundle.thresholds)

    # Label
    pred = int(np.argmax(P))
    pred_text = TARGET_MAP[pred]

    # Veto status
    veto_applied = (pconf_raw > (P_phys[0,2] + 1e-9))
    veto_reason = "b>1+ror or ror≥1" if veto_applied else None

    # Minimal human rationale (expand with SHAP/WASM later if needed)
    rationale = f"Calibrated probabilities with conformal set {S}. " \
                f"{'Physics veto applied; ' if veto_applied else ''}" \
                f"Threshold margins: {', '.join(f'{m:+.3f}' for m in margins)}."

    return PredictResult(
        proba=[float(x) for x in P],
        pred=pred, pred_text=pred_text,
        threshold_margins=margins,
        conformal_set=S,
        physics_veto_applied=veto_applied,
        veto_reason=veto_reason,
        rationale=rationale,
        top_features=[],
    )

@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    target = f"uploads/{file.filename}"
    with open(target, "wb") as f:
        f.write(await file.read())
    return {"status": "ok", "saved_as": target}

@app.post("/api/retrain")
def retrain(notes: str = Form("demo retrain")):
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S-UTC")
    log_file = f"registry/{timestamp}_train.log"
    os.makedirs("registry", exist_ok=True)
    with open(log_file, "w") as f:
        subprocess.run(["python", "step4_retrain.py", "--notes", notes],
                       stdout=f, stderr=f)
    return {
        "release": timestamp,
        "message": "Retrain complete",
        "log": log_file
    }

# ---------------- Upload & Retrain (Researcher mode) ----------------

def _require_retrain_token(token: Optional[str]):
    expected = os.getenv("RETRAIN_TOKEN")
    if not expected:
        # allow if explicitly enabled for demo
        if os.getenv("ALLOW_RETRAIN","0") != "1":
            raise HTTPException(403, "Retrain disabled. Set ALLOW_RETRAIN=1 or RETRAIN_TOKEN.")
        return True
    if token != expected:
        raise HTTPException(403, "Invalid token.")
    return True

@app.post("/api/upload")
def upload_csv(file: UploadFile = File(...),
               token: Optional[str] = Query(None, description="Bearer token (optional if ALLOW_RETRAIN=1)")):
    _require_retrain_token(token)
    # Save uploaded CSV to a temp path
    incoming = PROCESSED / f"upload_{int(time.time())}.csv"
    incoming.write_bytes(file.file.read())
    # Use Step-4 ingest (schema normalization + engineered features + pool handling)  
    try:
        summary = retrain_mod.ingest_csv(incoming)
    except Exception as e:
        raise HTTPException(400, f"Ingest failed: {e}")
    return {"status":"ok","ingest_summary": summary}

@app.post("/api/retrain")
def retrain(notes: Optional[str] = Form(""),
            discovery: Optional[bool] = Form(False),
            token: Optional[str] = Form(None)):
    _require_retrain_token(token)
    # Run Step-3 + snapshot under registry/v* + deltas + (auto Step-5)  
    try:
        summary = retrain_mod.run_step3_and_snapshot(notes=notes)
    except Exception as e:
        raise HTTPException(500, f"Training failed: {e}")

    # Optionally also run discovery CSV export (separate from Step-5 auto)
    disco_path = None
    if discovery:
        try:
            disco_path = retrain_mod.run_discovery_candidates(summary["release"])
        except Exception:
            disco_path = None

    # refresh in-memory bundle
    _ensure_bundle(reload=True)

    # Include diff vs previous, if STEP-4 saved it to SCOREBOARD.json
    scoreboard = (REGISTRY / "SCOREBOARD.json")
    deltas = json.loads(scoreboard.read_text()) if scoreboard.exists() else {}

    return {
        "status": "ok",
        "release": summary,
        "deltas_vs_previous": deltas.get("deltas_vs_previous"),
        "discovery_candidates_csv": disco_path
    }

# ------------------- Confusion Matrix Explorer -------------------
@app.get("/api/confusion")
def confusion(tau_conf: float=Query(0.5), tau_cand: float=Query(0.5), tau_fp: float=Query(0.5)):
    """
    Returns a confusion matrix computed on OOF/holdout if probs are available in reports.
    For now we simulate using class ratios from Step-3 report (display-only).
    """
    rep = _load_report()
    scores = rep.get("scores", {})
    # if you later add stored OOF probs, compute a real CM here
    kpis = {
        "oof": scores.get("oof", {}),
        "holdout": scores.get("holdout", {}),
        "thresholds": _load_thresholds_from_report(rep),
    }
    return kpis

# --- /api/universe: serve RA/Dec + physics for the 3D Universe ---
from fastapi import Query

@app.get("/api/universe")
def get_universe(
    limit: int = Query(5000, ge=1, le=50000),
    source: str = Query("auto", regex="^(auto|candidates|exounified)$")
):
    """
    Accurate sky positions for visualization.
    Priority:
      - auto: use Step-5 candidates if present; else exo_unified.parquet
      - candidates: force Step-5 CSV
      - exounified: force processed/exo_unified.parquet (with live inference)
    Returns rows with:
      mission, label, ra_deg, dec_deg, dist_pc, period, rade_Re, depth_ppm, p_exoplanet
    """
    import pandas as pd
    import numpy as np

    def _normalize_cols(df):
        # map many possible column names -> canonical
        colmap = {
            "ra_deg":    ["ra_deg","ra","RA","koi_ra","RAJ2000","RA_deg"],
            "dec_deg":   ["dec_deg","dec","DEC","koi_dec","DEJ2000","Dec_deg"],
            "period":    ["period","period_days","koi_period","tce_period","P"],
            "rade_Re":   ["rade_Re","radius_re","koi_prad","prad","Rp_Rearth"],
            "depth_ppm": ["depth_ppm","koi_depth","depth"],
            "mission":   ["mission","koi_mission","sector_mission","msn","source","catalog"],
            "label":     ["label","pred_text","disposition","pdisposition","koi_pdisposition","koi_disposition","tce_disposition"],
            "dist_pc":   ["dist_pc","dist","distance_pc","d_pc"]
        }
        out = {}
        for tgt, cands in colmap.items():
            for c in cands:
                if c in df.columns:
                    out[tgt] = c
                    break
        return out

    def _norm_mission(m):
        s = str(m).lower()
        if "tess" in s: return "TESS"
        if "k2"   in s: return "K2"
        return "Kepler"

    # Prefer registry/latest step5 if present, else reports/step5_candidates.csv
    cand = REGISTRY / "latest" / "reports" / "step5_candidates.csv"
    if not cand.exists():
        cand = REPORTS / "step5_candidates.csv"

    rows = []
    use_candidates = (source in ("auto","candidates")) and cand.exists()

    if use_candidates:
        df = pd.read_csv(cand)
        cols = _normalize_cols(df)
        # pick probability column
        prob_cols = [c for c in df.columns if c.lower() in ("p_exoplanet","p_conf","proba_confirmed")]
        pcol = prob_cols[0] if prob_cols else None

        keep = pd.DataFrame({
            "mission":    df[cols["mission"]] if "mission" in cols else "Kepler",
            "label":      df[cols["label"]]   if "label"   in cols else df.get("pred_text","Candidate"),
            "ra_deg":     pd.to_numeric(df[cols["ra_deg"]],    errors="coerce") if "ra_deg"    in cols else None,
            "dec_deg":    pd.to_numeric(df[cols["dec_deg"]],   errors="coerce") if "dec_deg"   in cols else None,
            "dist_pc":    pd.to_numeric(df[cols["dist_pc"]],   errors="coerce") if "dist_pc"   in cols else None,
            "period":     pd.to_numeric(df[cols["period"]],    errors="coerce") if "period"    in cols else None,
            "rade_Re":    pd.to_numeric(df[cols["rade_Re"]],   errors="coerce") if "rade_Re"   in cols else None,
            "depth_ppm":  pd.to_numeric(df[cols["depth_ppm"]], errors="coerce") if "depth_ppm" in cols else None,
            "p_exoplanet":pd.to_numeric(df[pcol],               errors="coerce") if pcol else None,
        })
        keep = keep.dropna(subset=["ra_deg","dec_deg"])
        rows = keep.head(limit).to_dict(orient="records")

    if (not rows) and (source in ("auto","exounified")):
        px = PROCESSED / "exo_unified.parquet"
        if not px.exists():
            raise HTTPException(404, "No step5 candidates and no exo_unified.parquet found.")

        df = pd.read_parquet(px)
        # ensure we have missions + coords
        if "mission" not in df.columns:
            df["mission"] = "Kepler"
        coord_mask = df.get("ra_deg").notna() & df.get("dec_deg").notna()
        sub = df.loc[coord_mask].copy()
        if sub.empty:
            raise HTTPException(404, "exo_unified.parquet has no RA/Dec.")

        # model inference to get calibrated p(Confirmed)
        bundle = _ensure_bundle()
        feature_cols = bundle.feats["feature_cols"]
        medians = bundle.feats["medians"]
        weights = bundle.feats["weights"]

        # build X (to model columns, with imputation)
        n = min(limit, len(sub))
        sub = sub.head(n)
        Xdf = pd.DataFrame([{c: sub[c].iloc[i] if c in sub.columns else None
                             for c in feature_cols} for i in range(n)])
        for c in feature_cols:
            Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce").fillna(medians.get(c, 0.0))
        X = Xdf.to_numpy().astype(float)

        Pw = np.zeros((n, 3))
        for k, mdl in bundle.models.items():
            w = float(weights.get(k, 0.0))
            if w <= 0: continue
            Pw += w * mdl.predict_proba(X)
        P_phys = physics_veto_adjust(Pw, Xdf)
        P_cal  = bundle.calibrator.predict_proba(P_phys)
        p_conf = P_cal[:, 2]

        # thresholds (class 2 = Confirmed, class 1 = Candidate)
        tau_conf = float(bundle.thresholds.get("2", 0.5))
        tau_cand = float(bundle.thresholds.get("1", 0.5))

        sub = sub.assign(
            p_exoplanet=p_conf,
            label=np.where(p_conf>=tau_conf, "Confirmed",
                           np.where(p_conf>=tau_cand, "Candidate", "False Positive"))
        )

        cols = _normalize_cols(sub)
        keep = pd.DataFrame({
            "mission":   sub[cols["mission"]] if "mission" in cols else "Kepler",
            "label":     sub["label"],
            "ra_deg":    pd.to_numeric(sub[cols["ra_deg"]],  errors="coerce") if "ra_deg"  in cols else None,
            "dec_deg":   pd.to_numeric(sub[cols["dec_deg"]], errors="coerce") if "dec_deg" in cols else None,
            "dist_pc":   pd.to_numeric(sub[cols["dist_pc"]], errors="coerce") if "dist_pc" in cols else None,
            "period":    pd.to_numeric(sub[cols["period"]],  errors="coerce") if "period"  in cols else None,
            "rade_Re":   pd.to_numeric(sub[cols["rade_Re"]], errors="coerce") if "rade_Re" in cols else None,
            "depth_ppm": pd.to_numeric(sub[cols["depth_ppm"]], errors="coerce") if "depth_ppm" in cols else None,
            "p_exoplanet": sub["p_exoplanet"].astype(float),
        })
        keep = keep.dropna(subset=["ra_deg","dec_deg"])
        rows = keep.to_dict(orient="records")

    if not rows:
        raise HTTPException(404, "No suitable rows found for universe.")

    # final tidy & typing
    for r in rows:
        r["mission"] = _norm_mission(r.get("mission","Kepler"))
        for k in ("ra_deg","dec_deg","period","rade_Re","depth_ppm","p_exoplanet","dist_pc"):
            if k in r:
                v = r[k]
                if v is None:
                    continue
                try:
                    v = float(v)
                    if math.isnan(v) or math.isinf(v):
                        v = None
                except:
                    v = None
                r[k] = v
    return rows

# ----------------------- Health & Root ------------------------
@app.get("/")
def root():
    return {"status":"ok","message":"Step-6 API up", "endpoints":[
        "/api/release","/api/metrics","/api/candidates","/api/explanations",
        "/api/predict","/api/upload","/api/retrain","/reports/*","/registry/*"
    ]}
