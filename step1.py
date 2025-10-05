import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# Config / constants
# =========================
OUTDIR = "processed"

LABEL_MAP_TO_3CLASS = {
    # KOI / K2
    "CONFIRMED": "Confirmed",
    "CANDIDATE": "Candidate",
    "FALSE POSITIVE": "False Positive",
    # TOI
    "CP": "Confirmed", "KP": "Confirmed",
    "PC": "Candidate", "APC": "Candidate",
    "FP": "False Positive",
}

# optional numeric encoding for modeling convenience
LABEL_TO_INT = {"False Positive": 0, "Candidate": 1, "Confirmed": 2}

# columns that we may winsorize (robust clipping of extreme outliers)
WINSOR_COLS = [
    "period","dur_h","depth_ppm","ror","rade_Re","a_over_rs","impact","ecc","incl_deg",
    "insol_S","eqt_K","teff_K","logg_cgs","feh_dex","rad_Rs","mass_Ms","age_Gyr",
    "ra_deg","dec_deg","pm_ra_masyr","pm_dec_masyr","dist_pc","mag_T","mag_Kepler",
    # engineered
    "depth_from_ror_ppm","depth_consistency","dur_frac","depth_snr","dur_snr","pm_tot"
]

TINY = 1e-12

# =========================
# Helper utilities
# =========================
# --- Label normalization & schema helpers ---
def norm_label_text(s):
    """Uppercase + trim + collapse spaces for robust mapping (e.g., 'Candidate ', 'confirmed')."""
    if pd.isna(s):
        return np.nan
    return str(s).upper().strip().replace("\u00A0", " ").replace("  ", " ")

def map_to_3class(raw_series):
    """Normalize text dispositions and map to 3-class labels + ints."""
    s = raw_series.map(norm_label_text)
    lbl = s.map(LABEL_MAP_TO_3CLASS)
    lbl_int = lbl.map(LABEL_TO_INT)
    return lbl, lbl_int

def enforce_schema(df, numeric_cols=WINSOR_COLS):
    """Ensure numeric columns are numeric, replace inf, and make star_id string."""
    df = df.copy()
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if "star_id" in df.columns:
        df["star_id"] = df["star_id"].astype(str)
    return df

def winsorize_df(df, cols, lower=0.1, upper=99.9):
    """Clip numeric columns to [p_lower, p_upper] percentiles for stability."""
    for c in cols:
        if c in df.columns:
            low, high = np.nanpercentile(df[c].astype(float), [lower, upper])
            df[c] = df[c].clip(lower=low, upper=high)
    return df

def add_uncertainty_bundle(df, base_col, err1_col, err2_col):
    """
    Build uncertainty features for a unified column.
    - base_col: unified value column (e.g., 'depth_ppm')
    - err1_col/err2_col: original err columns already mapped into df (same units)
    """
    if err1_col in df.columns and err2_col in df.columns:
        mean_name = f"{base_col}_err_mean"
        rel_name  = f"{base_col}_rel_err"
        miss_name = f"{base_col}_missing"
        df[mean_name] = (df[err1_col].abs() + df[err2_col].abs()) / 2
        df[rel_name]  = df[mean_name] / (df[base_col].abs().replace(0, np.nan))
        df[miss_name] = df[base_col].isna().astype(int)
    return df

def add_physics_features(df):
    # depth expected from (Rp/R*)^2
    if "ror" in df.columns:
        df["depth_from_ror_ppm"] = (df["ror"] ** 2) * 1e6
        if "depth_ppm" in df.columns:
            df["depth_consistency"] = (df["depth_ppm"] - df["depth_from_ror_ppm"]).abs() / df["depth_ppm"].replace(0, np.nan)
    # duration fraction
    if {"dur_h","period"} <= set(df.columns):
        df["dur_frac"] = (df["dur_h"]/24.0) / df["period"].replace(0, np.nan)
    # SNRs (only if uncertainty bundles exist)
    if {"depth_ppm","depth_ppm_err_mean"} <= set(df.columns):
        df["depth_snr"] = df["depth_ppm"] / df["depth_ppm_err_mean"].replace(0, np.nan)
    if {"dur_h","dur_h_err_mean"} <= set(df.columns):
        df["dur_snr"] = df["dur_h"] / df["dur_h_err_mean"].replace(0, np.nan)
    # proper motion magnitude
    if {"pm_ra_masyr","pm_dec_masyr"} <= set(df.columns):
        df["pm_tot"] = np.sqrt(df["pm_ra_masyr"]**2 + df["pm_dec_masyr"]**2)
    # flags
    if "ecc" in df.columns:
        df["ecc_flag"] = (df["ecc"] > 0).astype(int)
    if "impact" in df.columns:
        df["high_b"] = (df["impact"] > 0.8).astype(int)
    # Stellar radii / ratio consistency: Rp/Rs vs ror
    if {"rade_Re","rad_Rs","ror"} <= set(df.columns):
        rs_in_Re = df["rad_Rs"] * 109.2  # convert Rsun → Earth radii
        df["ror_from_radii"] = (df["rade_Re"] / rs_in_Re).replace([np.inf, -np.inf], np.nan)
        df["ror_consistency"] = (df["ror"] - df["ror_from_radii"]).abs() / df["ror"].replace(0, np.nan)
    return df

def sanitize_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# =========================
# External catalog cross-matching (Gaia, TIC)
# =========================
def crossmatch_catalogs(df, gaia_path=None, tic_path=None, radius_arcsec=2.0):
    """
    Cross-match stars with external catalogs (Gaia DR3, TIC).
    - df: your unified dataframe with RA/Dec
    - gaia_path: optional path to Gaia DR3 parquet/CSV (with ra, dec, phot_g_mean_mag, parallax, etc.)
    - tic_path: optional path to TIC parquet/CSV (with ra, dec, Tmag, stellar params, etc.)
    - radius_arcsec: match radius (default 2")
    Returns df with appended Gaia/TIC columns if available.
    """
    import numpy as np
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import pandas as pd

    df_out = df.copy()

    # Gaia crossmatch
    if gaia_path and Path(gaia_path).exists():
        gaia = pd.read_parquet(gaia_path) if gaia_path.endswith(".parquet") else pd.read_csv(gaia_path)
        coords_df   = SkyCoord(ra=df_out["ra_deg"].values*u.deg, dec=df_out["dec_deg"].values*u.deg)
        coords_gaia = SkyCoord(ra=gaia["ra"].values*u.deg, dec=gaia["dec"].values*u.deg)
        idx, sep2d, _ = coords_df.match_to_catalog_sky(coords_gaia)
        mask = sep2d.arcsec < radius_arcsec
        df_out.loc[mask, "gaia_source_id"] = gaia.iloc[idx[mask]]["source_id"].values
        df_out.loc[mask, "gaia_parallax"]  = gaia.iloc[idx[mask]]["parallax"].values
        df_out.loc[mask, "gaia_gmag"]      = gaia.iloc[idx[mask]]["phot_g_mean_mag"].values

    # TIC crossmatch
    if tic_path and Path(tic_path).exists():
        tic = pd.read_parquet(tic_path) if tic_path.endswith(".parquet") else pd.read_csv(tic_path)
        coords_df   = SkyCoord(ra=df_out["ra_deg"].values*u.deg, dec=df_out["dec_deg"].values*u.deg)
        coords_tic  = SkyCoord(ra=tic["ra"].values*u.deg, dec=tic["dec"].values*u.deg)
        idx, sep2d, _ = coords_df.match_to_catalog_sky(coords_tic)
        mask = sep2d.arcsec < radius_arcsec
        df_out.loc[mask, "tic_id"]       = tic.iloc[idx[mask]]["ID"].values
        df_out.loc[mask, "tic_Tmag"]     = tic.iloc[idx[mask]]["Tmag"].values
        df_out.loc[mask, "tic_st_rad"]   = tic.iloc[idx[mask]]["rad"].values
        df_out.loc[mask, "tic_st_mass"]  = tic.iloc[idx[mask]]["mass"].values

    return df_out

# =========================
# Mission-specific loaders
# =========================
def load_koi(path):
    # skip initial '#' metadata lines
    raw = pd.read_csv(path, comment="#", sep=",|\\t", engine="python")
    # if tab-separated (NASA exports often are), this handles both
    # ---- map to unified schema (+ bring err columns needed for uncertainty) ----
    keep = {
        "rowid": raw["rowid"],
        "star_id": raw["kepid"],
        "label_raw": raw["koi_disposition"],
        "period": raw["koi_period"],
        "period_err1": raw.get("koi_period_err1"),
        "period_err2": raw.get("koi_period_err2"),
        "t0_bjd": raw["koi_time0bk"].fillna(raw.get("koi_time0")),
        "t0_bjd_err1": raw.get("koi_time0bk_err1", raw.get("koi_time0_err1")),
        "t0_bjd_err2": raw.get("koi_time0bk_err2", raw.get("koi_time0_err2")),
        "dur_h": raw["koi_duration"],
        "dur_h_err1": raw.get("koi_duration_err1"),
        "dur_h_err2": raw.get("koi_duration_err2"),
        "depth_ppm": raw["koi_depth"],
        "depth_ppm_err1": raw.get("koi_depth_err1"),
        "depth_ppm_err2": raw.get("koi_depth_err2"),
        "ror": raw.get("koi_ror"),
        "ror_err1": raw.get("koi_ror_err1"),
        "ror_err2": raw.get("koi_ror_err2"),
        "rade_Re": raw.get("koi_prad"),
        "a_over_rs": raw.get("koi_dor"),
        "impact": raw.get("koi_impact"),
        "ecc": raw.get("koi_eccen"),
        "incl_deg": raw.get("koi_incl"),
        "insol_S": raw.get("koi_insol"),
        "eqt_K": raw.get("koi_teq"),
        "teff_K": raw.get("koi_steff"),
        "logg_cgs": raw.get("koi_slogg"),
        "feh_dex": raw.get("koi_smet"),
        "rad_Rs": raw.get("koi_srad"),
        "mass_Ms": raw.get("koi_smass"),
        "age_Gyr": raw.get("koi_sage"),
        "ra_deg": raw.get("ra"),
        "dec_deg": raw.get("dec"),
        "mag_Kepler": raw.get("koi_kepmag"),
    }
    df = pd.DataFrame(keep)
    # normalize labels to 3-class text, then numeric
    # normalize labels to 3-class text, then numeric
    df["label"], df["label_int"] = map_to_3class(df["label_raw"])
    # uncertainty bundles
    df = add_uncertainty_bundle(df, "period", "period_err1", "period_err2")
    df = add_uncertainty_bundle(df, "t0_bjd", "t0_bjd_err1", "t0_bjd_err2")
    df = add_uncertainty_bundle(df, "dur_h", "dur_h_err1", "dur_h_err2")
    df = add_uncertainty_bundle(df, "depth_ppm", "depth_ppm_err1", "depth_ppm_err2")
    df = add_uncertainty_bundle(df, "ror", "ror_err1", "ror_err2")
    # mission flags
    df["is_kepler"], df["is_k2"], df["is_tess"] = 1, 0, 0
    # ensure numeric dtypes for core columns
    df = sanitize_numeric(df, WINSOR_COLS)
    # engineered features
    df = add_physics_features(df)

    # --- Physics sanity (keep signals in a plausible regime) ---
    if "ror" in df:       
        df = df[(df["ror"].isna()) | (df["ror"] < 1.0)]
    if "impact" in df and "ror" in df:
        df = df[(df["impact"].isna()) | (df["ror"].isna()) | (df["impact"] <= 1.0 + df["ror"].fillna(0))]
    if "incl_deg" in df:  
        df = df[(df["incl_deg"].isna()) | ((df["incl_deg"] >= 0) & (df["incl_deg"] <= 90))]
    # Mission flag exclusivity check
    assert ((df["is_kepler"] + df["is_k2"] + df["is_tess"]) == 1).all(), \
        f"Mission flag inconsistency detected in {path}"
    return df

def load_toi(path):
    raw = pd.read_csv(path, comment="#", sep=",|\\t", engine="python")
    keep = {
        "rowid": raw["rowid"],
        "star_id": raw["tid"],
        "label_raw": raw["tfopwg_disp"],
        "period": raw.get("pl_orbper"),
        "period_err1": raw.get("pl_orbpererr1"),
        "period_err2": raw.get("pl_orbpererr2"),
        "t0_bjd": raw.get("pl_tranmid"),
        "t0_bjd_err1": raw.get("pl_tranmiderr1"),
        "t0_bjd_err2": raw.get("pl_tranmiderr2"),
        "dur_h": raw.get("pl_trandurh"),
        "dur_h_err1": raw.get("pl_trandurherr1"),
        "dur_h_err2": raw.get("pl_trandurherr2"),
        "depth_ppm": raw.get("pl_trandep"),
        "depth_ppm_err1": raw.get("pl_trandeperr1"),
        "depth_ppm_err2": raw.get("pl_trandeperr2"),
        "rade_Re": raw.get("pl_rade"),
        "insol_S": raw.get("pl_insol"),
        "eqt_K": raw.get("pl_eqt"),
        "teff_K": raw.get("st_teff"),
        "logg_cgs": raw.get("st_logg"),
        "rad_Rs": raw.get("st_rad"),
        "mag_T": raw.get("st_tmag"),
        "dist_pc": raw.get("st_dist"),
        "ra_deg": raw.get("ra"),
        "dec_deg": raw.get("dec"),
        "pm_ra_masyr": raw.get("st_pmra"),
        "pm_dec_masyr": raw.get("st_pmdec"),
    }
    df = pd.DataFrame(keep)
    df["label"], df["label_int"] = map_to_3class(df["label_raw"])
    # uncertainty bundles
    for base, e1, e2 in [
        ("period","period_err1","period_err2"),
        ("t0_bjd","t0_bjd_err1","t0_bjd_err2"),
        ("dur_h","dur_h_err1","dur_h_err2"),
        ("depth_ppm","depth_ppm_err1","depth_ppm_err2"),
    ]:
        df = add_uncertainty_bundle(df, base, e1, e2)
    df["is_kepler"], df["is_k2"], df["is_tess"] = 0, 0, 1
    df = sanitize_numeric(df, WINSOR_COLS)
    # engineered features
    df = add_physics_features(df)

    # --- Physics sanity (keep signals in a plausible regime) ---
    if "ror" in df:       
        df = df[(df["ror"].isna()) | (df["ror"] < 1.0)]
    if "impact" in df and "ror" in df:
        df = df[(df["impact"].isna()) | (df["ror"].isna()) | (df["impact"] <= 1.0 + df["ror"].fillna(0))]
    if "incl_deg" in df:  
        df = df[(df["incl_deg"].isna()) | ((df["incl_deg"] >= 0) & (df["incl_deg"] <= 90))]
    # Mission flag exclusivity check
    assert ((df["is_kepler"] + df["is_k2"] + df["is_tess"]) == 1).all(), \
        f"Mission flag inconsistency detected in {path}"
    return df

def load_k2(path):
    # First, try comma-separated with quotes respected
    try:
        raw = pd.read_csv(
            path,
            comment="#",
            sep=",",
            engine="python",
            quotechar='"',
            on_bad_lines="skip"  # skip problematic rows instead of failing
        )
    except Exception as e:
        print(f"[WARN] Standard CSV parse failed for {path}, retrying with tab delimiter. Error: {e}")
        raw = pd.read_csv(
            path,
            comment="#",
            sep="\t",
            engine="python",
            quotechar='"',
            on_bad_lines="skip"
        )
    #raw = pd.read_csv(path, comment="#", sep=",|\\t", engine="python")
    keep = {
        "rowid": raw["rowid"],
        "star_id": raw["hostname"].fillna(raw.get("epic_hostname")),
        "label_raw": raw["disposition"],
        "period": raw.get("pl_orbper"),
        "period_err1": raw.get("pl_orbpererr1"),
        "period_err2": raw.get("pl_orbpererr2"),
        "t0_bjd": raw.get("pl_tranmid"),
        "t0_bjd_err1": raw.get("pl_tranmiderr1"),
        "t0_bjd_err2": raw.get("pl_tranmiderr2"),
        "dur_h": raw.get("pl_trandur"),  # K2 duration column is 'pl_trandur' (hours)
        "dur_h_err1": raw.get("pl_trandurerr1"),
        "dur_h_err2": raw.get("pl_trandurerr2"),
        "depth_ppm": raw.get("pl_trandep"),
        "depth_ppm_err1": raw.get("pl_trandeperr1"),
        "depth_ppm_err2": raw.get("pl_trandeperr2"),
        "ror": raw.get("pl_ratror"),
        "ror_err1": raw.get("pl_ratrorerr1"),
        "ror_err2": raw.get("pl_ratrorerr2"),
        "a_over_rs": raw.get("pl_ratdor"),
        "a_over_rs_err1": raw.get("pl_ratdorerr1"),
        "a_over_rs_err2": raw.get("pl_ratdorerr2"),
        "impact": raw.get("pl_imppar"),
        "ecc": raw.get("pl_orbeccen"),
        "incl_deg": raw.get("pl_orbincl"),
        "rade_Re": raw.get("pl_rade"),
        "insol_S": raw.get("pl_insol"),
        "eqt_K": raw.get("pl_eqt"),
        "teff_K": raw.get("st_teff"),
        "logg_cgs": raw.get("st_logg"),
        "feh_dex": raw.get("st_met"),
        "rad_Rs": raw.get("st_rad"),
        "mass_Ms": raw.get("st_mass"),
        "age_Gyr": raw.get("st_age"),
        "ra_deg": raw.get("ra"),
        "dec_deg": raw.get("dec"),
        "pm_ra_masyr": raw.get("sy_pmra"),
        "pm_dec_masyr": raw.get("sy_pmdec"),
        "dist_pc": raw.get("sy_dist"),
        "mag_T": raw.get("sy_tmag"),
        "mag_Kepler": raw.get("sy_kepmag"),
    }
    df = pd.DataFrame(keep)
    df["label"], df["label_int"] = map_to_3class(df["label_raw"])
    # uncertainty bundles
    for base, e1, e2 in [
        ("period","period_err1","period_err2"),
        ("t0_bjd","t0_bjd_err1","t0_bjd_err2"),
        ("dur_h","dur_h_err1","dur_h_err2"),
        ("depth_ppm","depth_ppm_err1","depth_ppm_err2"),
        ("ror","ror_err1","ror_err2"),
        ("a_over_rs","a_over_rs_err1","a_over_rs_err2"),
    ]:
        df = add_uncertainty_bundle(df, base, e1, e2)
    df["is_kepler"], df["is_k2"], df["is_tess"] = 0, 1, 0
    df = sanitize_numeric(df, WINSOR_COLS)
    # engineered features
    df = add_physics_features(df)

    # --- Physics sanity (keep signals in a plausible regime) ---
    if "ror" in df:       
        df = df[(df["ror"].isna()) | (df["ror"] < 1.0)]
    if "impact" in df and "ror" in df:
        df = df[(df["impact"].isna()) | (df["ror"].isna()) | (df["impact"] <= 1.0 + df["ror"].fillna(0))]
    if "incl_deg" in df:  
        df = df[(df["incl_deg"].isna()) | ((df["incl_deg"] >= 0) & (df["incl_deg"] <= 90))]
    # Mission flag exclusivity check
    assert ((df["is_kepler"] + df["is_k2"] + df["is_tess"]) == 1).all(), \
        f"Mission flag inconsistency detected in {path}"
    return df

# =========================
# Master ETL
# =========================
def build_unified_dataset(koi_csv, toi_csv, k2_csv, outdir=OUTDIR, winsorize=True):
    Path(outdir).mkdir(exist_ok=True)

    df_koi = load_koi(koi_csv)
    df_toi = load_toi(toi_csv)
    df_k2  = load_k2(k2_csv)

    # Clip implausible values per our QA rules
    for df in (df_koi, df_toi, df_k2):
        if "period" in df:   df = df[df["period"]   > 0]
        if "dur_h" in df:    df = df[df["dur_h"]    > 0]
        if "depth_ppm" in df:df = df[df["depth_ppm"]> 0]
        if "rad_Rs" in df:   df = df[df["rad_Rs"]   > 0]

    if winsorize:
        df_koi = winsorize_df(df_koi, WINSOR_COLS)
        df_toi = winsorize_df(df_toi, WINSOR_COLS)
        df_k2  = winsorize_df(df_k2,  WINSOR_COLS)

    # Save mission files
    df_koi.to_parquet(Path(outdir)/"kepler.parquet", index=False)
    df_toi.to_parquet(Path(outdir)/"tess.parquet",   index=False)
    df_k2.to_parquet(Path(outdir)/"k2.parquet",      index=False)

    # Merge missions
    df_all = pd.concat([df_koi, df_toi, df_k2], axis=0, ignore_index=True)

    df_all["star_id"] = df_all["star_id"].astype(str)

    # Optional: cross-match with Gaia/TIC catalogs for bonus astrophysics points
    gaia_file = "/Users/lalithbhima/NASA/gaia_dr3.parquet"
    tic_file  = "/Users/lalithbhima/NASA/tic8.parquet"
    df_all = crossmatch_catalogs(df_all, gaia_path=gaia_file, tic_path=tic_file, radius_arcsec=2.0)
    # Log unmatched rows for catalog cross-matching
    unmatched_gaia = df_all["gaia_parallax"].isna().sum() if "gaia_parallax" in df_all else 0
    unmatched_tic  = df_all["tic_Tmag"].isna().sum() if "tic_Tmag" in df_all else 0
    with open(Path(outdir)/"etl_unmapped_log.txt", "a") as f:
        f.write(f"Gaia unmatched rows: {unmatched_gaia}\n")
        f.write(f"TIC unmatched rows: {unmatched_tic}\n")


    # Drop rows with unmapped/unknown labels (prevents Step 3 crashes)
    before = len(df_all)
    df_all = df_all[df_all["label_int"].notna()].copy()
    dropped = before - len(df_all)
    if dropped > 0:
        Path(outdir).mkdir(exist_ok=True)
        df_bad = df_all[df_all["label_int"].isna()] if "label_int" in df_all.columns else pd.DataFrame()
        # Log a small report of what was dropped (if any were encountered earlier)
        with open(Path(outdir)/"etl_unmapped_log.txt", "a") as f:
            f.write(f"Dropped {dropped} rows with unmapped labels.\n")

    # Optional: remove obvious duplicates (same star_id, period, t0_bjd)
    dup_keys = [c for c in ["star_id", "period", "t0_bjd"] if c in df_all.columns]
    if dup_keys:
        before_dups = len(df_all)
        df_all = df_all.drop_duplicates(subset=dup_keys, keep="first")
        removed_dups = before_dups - len(df_all)
        if removed_dups > 0:
            with open(Path(outdir)/"etl_unmapped_log.txt", "a") as f:
                f.write(f"Removed {removed_dups} duplicate rows on keys {dup_keys}.\n")

    # Final anti-leakage: keep only unified + engineered columns (no discovery/vetting meta present here)
    # (we already never brought meta columns in, so we’re safe)

    df_all.to_parquet(Path(outdir)/"exo_unified.parquet", index=False)
    # Also save a small schema preview for UI/debug
    df_all.head(50).to_csv(Path(outdir)/"exo_unified_preview.csv", index=False)
    # Save Gaia/TIC-augmented preview too
    cols_for_preview = [c for c in df_all.columns if c.startswith("gaia_") or c.startswith("tic_")] + ["star_id","period","ror"]
    df_all[cols_for_preview].head(50).to_csv(Path(outdir)/"exo_unified_preview_catalog.csv", index=False)
    # =========================
    # Export schema metadata
    # =========================
    schema = {
        "raw_features": [c for c in df_all.columns if c not in [
            "label","label_int","is_kepler","is_k2","is_tess"
        ] and not c.endswith(("_err_mean","_rel_err","_missing"))]
        + ["gaia_parallax","gaia_gmag","tic_Tmag","tic_st_rad","tic_st_mass"],  # <--- ADDITIONAL CATALOG COLUMNS
        "uncertainty_bundles": [c for c in df_all.columns if c.endswith(("_err_mean","_rel_err","_missing"))],
        "engineered_features": [c for c in [
            "depth_from_ror_ppm","depth_consistency","dur_frac",
            "depth_snr","dur_snr","pm_tot","ecc_flag","high_b"
        ] if c in df_all.columns],
        "mission_flags": ["is_kepler","is_k2","is_tess"],
        "label_columns": ["label","label_int"]
    }
    import yaml
    with open(Path(outdir)/"features.yaml", "w") as f:
        yaml.dump(schema, f)
    print("📝 Feature schema exported → features.yaml")
    # =========================
    # Sanity check + reporting
    # =========================
    def summarize(df, name):
        if "label" in df.columns:
            counts = df["label"].value_counts().to_dict()
        else:
            counts = {}
        print(f"\n--- {name} ---")
        print("Rows:", len(df))
        print("Class counts:", counts)
        # show physics features if available
        for col in ["depth_consistency","dur_frac","depth_snr","dur_snr","pm_tot"]:
            if col in df.columns:
                print(f"{col} (non-null):", df[col].notna().sum())
        print("----------------------")

    # Print summaries for each dataset
    summarize(df_koi, "Kepler (KOI)")
    summarize(df_toi, "TESS (TOI)")
    summarize(df_k2,  "K2")
    summarize(df_all, "Unified (All Missions)")
    # =========================
    # Baseline sanity plots
    # =========================
    import matplotlib.pyplot as plt

    qc_cols = ["period","ror","dur_frac","depth_consistency"]
    plt.figure(figsize=(12,8))
    for i,col in enumerate(qc_cols,1):
        if col in df_all.columns:
            plt.subplot(2,2,i)
            df_all[col].dropna().hist(bins=50)
            plt.title(col)
    plt.tight_layout()
    plt.savefig(Path(outdir)/"step1_qc.pdf")
    plt.close()
    print("📊 QC plots saved → step1_qc.pdf")
    # Save report to file for demo
    report = {
        "Kepler": {
            "rows": len(df_koi),
            "class_counts": df_koi["label"].value_counts().to_dict(),
        },
        "TESS": {
            "rows": len(df_toi),
            "class_counts": df_toi["label"].value_counts().to_dict(),
        },
        "K2": {
            "rows": len(df_k2),
            "class_counts": df_k2["label"].value_counts().to_dict(),
        },
        "Unified": {
            "rows": len(df_all),
            "class_counts": df_all["label"].value_counts().to_dict(),
        }
    }
    import json
    with open(Path(outdir)/"etl_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return df_all

def process_csv(csv_path, mission, outdir=OUTDIR, append=True):
    """
    Process a new uploaded CSV into the unified schema for retraining.
    - csv_path: path to the uploaded CSV
    - mission: one of {"kepler","k2","tess"}
    - outdir: processed directory
    - append: if True, append to existing exo_unified.parquet
    """
    Path(outdir).mkdir(exist_ok=True)

    if mission.lower() == "kepler":
        df_new = load_koi(csv_path)
        fname = "kepler_new.parquet"
    elif mission.lower() == "k2":
        df_new = load_k2(csv_path)
        fname = "k2_new.parquet"
    elif mission.lower() == "tess":
        df_new = load_toi(csv_path)
        fname = "tess_new.parquet"
    else:
        raise ValueError("mission must be one of {'kepler','k2','tess'}")

    # Winsorize to keep physics stable
    df_new = winsorize_df(df_new, WINSOR_COLS)

    # --- Guardrail: drop unmapped labels + log + dedupe ---
    before_new = len(df_new)
    df_new = df_new[df_new["label_int"].notna()].copy()
    dropped_new = before_new - len(df_new)
    if dropped_new > 0:
        with open(Path(outdir)/"etl_unmapped_log.txt", "a") as f:
            f.write(f"{mission}: dropped {dropped_new} rows with unmapped labels from {csv_path}\n")

    dup_keys = [c for c in ["star_id", "period", "t0_bjd"] if c in df_new.columns]
    if dup_keys:
        df_new = df_new.drop_duplicates(subset=dup_keys, keep="first")

    # Save mission-specific parquet
    df_new.to_parquet(Path(outdir)/fname, index=False)

    if append and (Path(outdir)/"exo_unified.parquet").exists():
        df_all = pd.read_parquet(Path(outdir)/"exo_unified.parquet")
        df_all = pd.concat([df_all, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_parquet(Path(outdir)/"exo_unified.parquet", index=False)
    print(f"✅ Processed {len(df_new)} rows for {mission}, unified dataset now {len(df_all)} rows.")
    return df_new

# Example (adjust if running outside this environment):
# build_unified_dataset(
#     "/mnt/data/kelper_objects_of_interest.csv",
#     "/mnt/data/TESS_objects_of_interest.csv",
#     "/mnt/data/k2_planets_and_candidates.csv"
# )

def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Step 1 engineered features to any unified DataFrame.
    Used in Step 4 retraining ingestion.
    """
    # Enforce numeric schema
    df = enforce_schema(df, WINSOR_COLS)

    # Add physics-aware features
    df = add_physics_features(df)

    # Add uncertainty bundles if error columns exist
    bundles = [
        ("period","period_err1","period_err2"),
        ("t0_bjd","t0_bjd_err1","t0_bjd_err2"),
        ("dur_h","dur_h_err1","dur_h_err2"),
        ("depth_ppm","depth_ppm_err1","depth_ppm_err2"),
        ("ror","ror_err1","ror_err2"),
        ("a_over_rs","a_over_rs_err1","a_over_rs_err2"),
    ]
    for base, e1, e2 in bundles:
        if base in df.columns:
            df = add_uncertainty_bundle(df, base, e1, e2)

    # Winsorize (clip extremes) for physics stability
    df = winsorize_df(df, WINSOR_COLS)

    # Physics sanity guardrails
    if "period" in df:   df = df[df["period"] > 0]
    if "dur_h" in df:    df = df[df["dur_h"] > 0]
    if "depth_ppm" in df:df = df[df["depth_ppm"] > 0]
    if "rad_Rs" in df:   df = df[df["rad_Rs"] > 0]
    if "incl_deg" in df: df = df[(df["incl_deg"].isna()) | ((df["incl_deg"] >= 0) & (df["incl_deg"] <= 90))]
    if "ror" in df:      df = df[(df["ror"].isna()) | (df["ror"] < 1.0)]
    if "impact" in df and "ror" in df:
        df = df[(df["impact"].isna()) | (df["ror"].isna()) | (df["impact"] <= 1.0 + df["ror"].fillna(0))]

    return df

if __name__ == "__main__":
    # Paths to your raw CSV datasets
    koi_path = "/Users/lalithbhima/NASA/kelper_objects_of_interest.csv"
    toi_path = "/Users/lalithbhima/NASA/TESS_objects_of_interest.csv"
    k2_path  = "/Users/lalithbhima/NASA/k2_planets_and_candidates.csv"

    print("🔭 Building unified exoplanet dataset (KOI + TOI + K2)...")
    df_all = build_unified_dataset(koi_path, toi_path, k2_path, outdir="processed")

    print("✅ Done! Unified dataset shape:", df_all.shape)
    print("Columns:", list(df_all.columns)[:25], "...")
    print("Sample rows:\n", df_all.head())
