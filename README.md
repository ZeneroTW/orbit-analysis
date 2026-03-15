# 🛰️ Satellites & Space Debris - Orbit Analysis & Prediction

> 🇷🇺 [Читать на русском](README.ru.md)

A two-part data science project on satellites and space debris orbiting Earth. The dataset is sourced from [Space-Track.org](https://www.space-track.org) via API and covers orbital parameters of thousands of objects catalogued as of November 2021.

**Part 1** covers exploratory data analysis and data preparation. **Part 2** builds regression models to predict the orbital period.

---

## Dataset

**Source:** [Kaggle - Satellites and Debris in Earth's Orbit](https://www.kaggle.com/datasets/kandhalkhandeka/satellites-and-debris-in-earths-orbit)  
**File:** `space_decay.csv`

### Key Features

| Feature | Description |
|---|---|
| `OBJECT_TYPE` | Object type: PAYLOAD, DEBRIS, ROCKET BODY |
| `INCLINATION` | Orbital inclination (degrees) |
| `ECCENTRICITY` | Orbital eccentricity |
| `APOAPSIS` | Apogee altitude (km) |
| `PERIAPSIS` | Perigee altitude (km) |
| `SEMIMAJOR_AXIS` | Semi-major axis (km) |
| `PERIOD` | Orbital period (minutes) — **target variable** |
| `MEAN_MOTION` | Mean motion (revolutions/day) |
| `BSTAR` | Ballistic drag coefficient |
| `COUNTRY_CODE` | Launching country |
| `RCS_SIZE` | Radar cross-section size: SMALL / MEDIUM / LARGE |
| `LAUNCH_DATE` | Launch year |
| `NORAD_CAT_ID` | NORAD catalog number |

---

## Part 1 — Exploratory Data Analysis (`01_eda.ipynb`)

### Pipeline

1. **Data Loading** — download via `kagglehub`, drop technical/non-informative columns, parse `EPOCH` as datetime
2. **Preprocessing** — fill missing values (mode for categoricals, mean for numerics), detect duplicates and physical anomalies
3. **EDA** — distribution histograms, correlation heatmap, scatter plots, top-10 countries, object type breakdown
4. **Feature Engineering** — `ALT_MEAN` (mean orbital altitude), `ORBIT_CLASS` (LEO / MEO / GEO)
5. **Outlier Handling** — remove objects with `ALT_MEAN > 50,000 km`
6. **Categorical Encoding** — One-Hot Encoding for `RCS_SIZE`, `OBJECT_TYPE`, `SITE`
7. **Feature Selection** — keep features with |correlation to PERIOD| > 0.3
8. **Export** — save cleaned dataset to `orbits.csv`

---

## Part 2 — Machine Learning (`02_modeling.ipynb`)

### Goal
Predict the orbital period (`PERIOD`) of an object based on its orbital parameters.

### Pipeline

1. **Data Loading** — read `orbits.csv` produced by Part 1
2. **Train/Test Split** — 80% train / 20% test, `random_state=42`
3. **Baseline Models** — Linear Regression, Random Forest, Gradient Boosting
4. **Model Comparison** — MAE, RMSE, R² on test set; Actual vs Predicted plot
5. **Hyperparameter Tuning** — `RandomizedSearchCV` for Random Forest and Gradient Boosting
6. **Evaluation** — train vs test metrics to diagnose overfitting; updated comparison table
7. **Feature Importance** — which orbital parameters matter most for prediction
8. **Experiments** — 70/30 split, feature scaling (StandardScaler)
9. **Cross-Validation** — 5-fold CV RMSE for all models
10. **Sensitivity Analysis** — effect of `n_estimators` (RF) and `learning_rate` (GB) on quality
11. **Model Export** — save best models with `joblib`

---

## Tech Stack

- **Python 3**
- **Pandas, NumPy** — data manipulation
- **Matplotlib, Seaborn** — visualizations
- **Scikit-learn** — ML models, tuning, evaluation
- **KaggleHub** — dataset download
- **Joblib** — model serialization
- **Google Colab** — runtime environment

---

## Getting Started

1. Open notebooks in [Google Colab](https://colab.research.google.com/)
2. Make sure your Kaggle API token (`kaggle.json`) is configured
3. Run `01_eda.ipynb` first — it produces `orbits.csv`
4. Then run `02_modeling.ipynb`

---

## Project Structure

```
├── 01_eda.ipynb               # Part 1: EDA & data preparation
├── 02_modeling.ipynb          # Part 2: ML models & evaluation
├── orbits.csv                 # Cleaned dataset (output of Part 1)
├── best_rf.joblib             # Best Random Forest model
├── best_gb.joblib             # Best Gradient Boosting model
├── linear_baseline.joblib     # Linear Regression baseline
├── README.md                  # This file (English)
└── README.ru.md               # Russian version
```

---

## Key Findings

The vast majority of Earth's orbital objects are debris (DEBRIS), concentrated mainly in Low Earth Orbit (LEO). Russia/CIS, the USA, and China lead by object count. The orbital period (`PERIOD`) correlates most strongly with the semi-major axis (`SEMIMAJOR_AXIS`) and mean altitude — consistent with Kepler's Third Law. In Part 2, ensemble models (Random Forest, Gradient Boosting) significantly outperformed Linear Regression, with tuned models achieving R² close to 1.0.

---

*Academic project — RTU MIREA, Data Analysis practical assignment.*
