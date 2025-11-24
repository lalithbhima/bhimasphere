#  BhimaSphere: Hunting for Exoplanets with AI

**Team Members:**  
 1. Lalithendra Reddy Bhima  
 2. Bhavika Bhima  

**Challenge Theme:**  
 *A World Away – Hunting for Exoplanets with AI*  
**Event:** NASA Space Apps Challenge 2025  

---

##  Overview

**BhimaSphere** is an AI-powered platform that detects and classifies exoplanets across NASA missions using advanced ensemble and deep learning models.  
Our goal is to make exoplanet discovery **faster, more interpretable, and universally accessible** through an interactive 3D Universe that visualizes each planet’s position, probability, and confidence.

> “From raw telescope data to planetary discovery — all in one AI-powered Universe.”

---

##  Core Idea

BhimaSphere automates the process of identifying exoplanets from NASA’s **Kepler**, **K2**, and **TESS** missions by combining:
- **Physics-aware feature engineering**
- **AI ensemble learning (LightGBM, XGBoost, CatBoost)**
- **Deep neural networks (TabTransformer, MLP, CNN + BiLSTM)**
- **Explainability (SHAP + probability calibration)**
- **Dynamic retraining** with user-uploaded datasets  

The system generalizes across missions — enabling planetary candidates, confirmed exoplanets, and false positives to be analyzed in a unified framework.

---

##  How It Works

###  Step 1 — Data Preparation
- **Datasets:**  
  - NASA *Kepler Objects of Interest (KOI)*  
  - NASA *K2 Planets & Candidates*  
  - NASA *TESS Objects of Interest (TOI)*  

- **Physics-based features extracted:**
  - Orbital period, radius ratio, transit depth/duration
  - Inclination, eccentricity, stellar radius/temperature, SNR ratios
  - Derived relationships: depth consistency, duration residuals, etc.

- **Data cleaning:**  
  - Drop leakage columns  
  - Normalize units, handle NaNs, encode categoricals  
  - Mission harmonization → *Unified Cross-Mission Schema*

---

###  Step 2 — AI/ML Pipeline

| Stage | Component | Purpose |
|:------|:-----------|:--------|
| **Feature Extraction** | TSFRESH time-series features | Converts light-curve signals to numerical patterns |
| **Tree Ensembles** | LightGBM / XGBoost / CatBoost | Captures non-linear relationships |
| **Deep Learning** | TabTransformer + MLP | Learns latent physics-aware embeddings |
| **Sequential Model** | CNN + BiLSTM | Detects transit dips & periodicity in brightness |
| **Meta-Stacker** | Logistic Regression | Blends neural + ensemble outputs for calibrated results |

 **Classification:**  
`Confirmed`, `Candidate`, `False Positive`

 **Calibration:**  
Temperature scaling → trustworthy probability estimates  

 **Explainability:**  
Global + local SHAP values visualize which physics features influenced each detection.

---

###  Step 3 — Interactive 3D Universe

Built with **Three.js + React + WebGL**, the 3D Universe visualizes:
- Every detected planet (RA, Dec, Distance)
- Color-coded by **Mission / Probability / Label**
- **Hover** → shows full planetary metrics  
- **Click** → open info panel with:
  - Mission, classification, p(exoplanet), radius, period, confidence
  - SHAP metrics & calibration scores  

**Modes:**
-  *Novice Mode* – guided exploration with explanations  
-  *Researcher Mode* – upload CSVs, retrain models, explore live calibration  

**Features:**
- Dynamic retraining (Steps 1–5 pipeline)  
- Screenshot + Export CSV  
- Threshold slider (τ Confirmed)  
- Color blending & base size controls  
- Tooltip + 3D rotation + zoom navigation  

---

##  Model Architecture Diagram

![Model Architecture](https://github.com/lalithbhima/bhimasphere/blob/main/docs/model_architecture.png)


> The model combines **tree ensembles**, **deep learning**, and **stacking** for mission-independent, physics-consistent exoplanet discovery.

---

##  Physics Integration

Every stage of BhimaSphere integrates real astrophysical principles:

| Physics Parameter | Role in AI Model |
|:------------------|:----------------|
| **Transit depth & duration** | Determines planetary size & orbit |
| **Radius ratio (Rp/R★)** | Used to verify light curve consistency |
| **Stellar temperature & luminosity** | Encodes star type (influences flux) |
| **Orbital period (days)** | Links to Kepler’s Third Law features |
| **Inclination & eccentricity** | Identify geometric transit likelihood |
| **Signal-to-noise ratios (SNR)** | Confidence measure in detection |
| **Photometric density & derived gravity** | Physics-informed validation metric |

These are automatically **normalized, scaled, and embedded** into the model for physically meaningful learning.

---

##  Results

| Metric | Holdout | ROC-AUC | PR-AUC |
|:-------|:--------|:--------|:--------|
| Kepler | ✅ 0.900 | 0.815 |
| K2 | ✅ 0.887 | 0.802 |
| TESS | ✅ 0.864 | 0.780 |
| Unified | 🌍 **0.900 ROC / 0.815 PR-AUC (Holdout)** |

**Threshold (τ Confirmed):** 0.423  
Balanced for high recall and low false-positive rate.

---

##  Running BhimaSphere

### Backend (FastAPI)
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 7860
```
### Frontend (React + Three.js)
```bash
cd nasa-exo-ui
npm install
npm run dev
```
Then open http://localhost:5173/universe

---

##  Folder Structure

```bash
NASA/
├── models/               # Trained ensemble + neural weights
├── nasa-exo-ui/          # Frontend (React + Three.js)
│   ├── components/
│   ├── pages/
│   └── data/
│       ├── kepler_objects_of_interest.json
│       ├── k2_planets_and_candidates.json
│       └── TESS_objects_of_interest.json
├── step1.py              # Data prep & feature engineering
├── step2_train.py        # Mission-specific model training
├── step3_unified_model.py # Cross-mission ensemble stack
├── step4_retrain.py      # Dynamic retraining handler
└── step5_explain_discovery.py # SHAP & candidate discovery
```

---

##  Demo & Presentation

-  **Demo Video:** [Watch on YouTube](https://youtu.be/a8_UtA3NojI)  
-  **Presentation Slides:** https://www.canva.com/design/DAG02SIRMBg/YxXyYPrEOYJq-cmJbcq2_Q/view

---

##  Future Enhancements

- Integrate **raw light curve ingestion** with automated folding  
- Deploy on **NASA's Exoplanet Archive sandbox** for public use  
- Add **Spectral classification** and **TESS FFI auto-curation**  
- Incorporate **reinforcement-based retraining** for live discoveries  

---

##  Acknowledgments

This project was developed for the **2025 NASA Space Apps Challenge** using publicly available data from:

- **NASA Exoplanet Archive**  
- **Kepler, K2, and TESS Missions**  
- **MNRAS (Malik et al., 2022)** and **Electronics (2024)** reference architectures  

---








