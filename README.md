# English Premier League match outcome prediction (2015–2025)

Predict EPL match results (**home win**, **draw**, **away win**) from historical match statistics. This repository includes:

- **Jupyter notebooks** — exploratory analysis, statistics, and model evaluation  
- **`app/ml_core.py`** — training and inference aligned with the notebooks  
- **Two browser UIs** — **FastAPI** + static HTML (`web/`) and **Streamlit** (`streamlit_app.py`), both reading the same trained **`bundle.joblib`**

---

## Contents

- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Dataset](#dataset)
- [Modeling approach](#modeling-approach)
- [Install](#install)
- [Train the model bundle](#train-the-model-bundle)
- [Run FastAPI](#run-fastapi)
- [Run Streamlit](#run-streamlit)
- [HTTP API](#http-api)
- [Environment variables](#environment-variables)
- [Static UI and the API](#static-ui-and-the-api)
- [Troubleshooting](#troubleshooting)
- [Git](#git)
- [Keeping notebooks and code in sync](#keeping-notebooks-and-code-in-sync)
- [Data and license](#data-and-license)

---

## Quick start

```powershell
cd "path\to\this\folder"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place **`Football_Dataset_2015_2025.csv`** in the project root (same folder as `app/`), then:

```powershell
python scripts/train_model.py
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- Open **http://127.0.0.1:8000/** for the HTML UI.  
- Or run **`streamlit run streamlit_app.py`** (from the same folder) for the Streamlit UI — often **http://localhost:8501**.

---

## Repository layout

| Path | Description |
|------|-------------|
| `EPL_Project_Notebook.ipynb` | Full workflow: load data, EDA, statistics, train/evaluate models. |
| `EPL_Group5_Exam_Notebook.ipynb` | Same analysis with a group title page for submissions. |
| `app/main.py` | FastAPI app factory: lifespan, CORS, static files; `/` serves `web/index.html`. |
| `app/api/routes.py` | REST routes under `/api`. |
| `app/schemas.py` | Pydantic request/response models. |
| `app/config.py` | Paths and CORS from environment (`get_settings()`). |
| `app/services/bundle.py` | Thread-safe load/cache of `bundle.joblib` (invalidates on file mtime). |
| `app/ml_core.py` | EPL filter, features, preprocessing, training, `predict_row`. |
| `app/constants.py`, `app/labels.py` | Model catalog (`rf` / `log` / `xgb`) and human-readable outcome labels. |
| `web/` | Static UI: `index.html`, `styles.css`, `app.js`. |
| `streamlit_app.py` | Streamlit dashboard (same predictions as the API). |
| `scripts/train_model.py` | CLI to fit models and write `artifacts/bundle.joblib`. |
| `requirements.txt` | Python dependencies (notebooks + FastAPI + Streamlit + ML stack). |
| `artifacts/` | Generated models (see [.gitignore](.gitignore); only `artifacts/.gitkeep` is tracked by default). |

---

## Dataset

| Item | Detail |
|------|--------|
| **Expected file** | `Football_Dataset_2015_2025.csv` at the project root (or pass `--csv` to the train script). |
| **Raw size** | About 3,000 rows across several competitions. |
| **Modeling subset** | Rows where **`Competition == "Premier League"`** (typically ~591 matches, 2015–2025). |

The CSV is **gitignored** (`*.csv`) so it is not pushed by default. Adjust `.gitignore` if your policy requires committing data (not recommended for large or restricted datasets).

---

## Modeling approach

| Item | Detail |
|------|--------|
| **Target** | `Winner`: `Home Team`, `Draw`, `Away Team` (exact CSV strings). |
| **Features** | Home/away team (one-hot), year, possession, shots, corners, fouls; **month** and **day-of-week** from `Date`. |
| **Models** | Random Forest (400 trees, `class_weight='balanced'`), Logistic Regression (`max_iter=5000`), XGBoost (multiclass softmax). Each uses the same sklearn `Pipeline` (imputer + one-hot for categorical teams). |
| **Split** | 80% / 20% train/test, stratified, `random_state=42`. |

Random Forest usually leads on this dataset; exact scores depend on your CSV.

---

## Install

**Python 3.10+** is recommended (3.11 or 3.12 are ideal; 3.13 works if `xgboost` installs cleanly for your platform).

```powershell
pip install -r requirements.txt
```

---

## Train the model bundle

Run from the **repository root** (the directory that contains `app/` and `scripts/`):

```powershell
python scripts/train_model.py
```

| Option | Meaning |
|--------|---------|
| `--csv PATH` | Input CSV (default: `./Football_Dataset_2015_2025.csv`). |
| `--out PATH` | Output joblib path (default: value of `EPL_ARTIFACT_PATH`, or `artifacts/bundle.joblib`). |

**Outputs**

- **`artifacts/bundle.joblib`** — fitted pipelines, label encoder, team list, per-model test metrics.  
- **`artifacts/bundle_metrics.json`** — accuracy / macro-F1 / weighted-F1 on the hold-out set.

After you replace `bundle.joblib`, the FastAPI **`BundleStore`** reloads it on the next request (file **mtime** change). You usually **do not** need to restart Uvicorn.

---

## Run FastAPI

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

| URL | Purpose |
|-----|---------|
| http://127.0.0.1:8000/ | Static predictor UI |
| http://127.0.0.1:8000/docs | Swagger UI — schemas and “Try it out” |
| http://127.0.0.1:8000/redoc | ReDoc |

If the bundle is missing, **`GET /api/meta`** returns **503**; the HTML status line explains that you need to run `scripts/train_model.py` first.

---

## Run Streamlit

```powershell
streamlit run streamlit_app.py
```

Use the URL Streamlit prints (commonly **http://localhost:8501**). The sidebar shows hold-out metrics stored in the bundle; the main area mirrors the API inputs (teams, date, year, match stats).

---

## HTTP API

Base path: **`/api`**. All responses are JSON unless noted.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | `{ ok, model_loaded, artifact }` — whether `bundle.joblib` exists at the configured path. |
| GET | `/api/` | Short message pointing to `/docs`. |
| GET | `/api/meta` | Teams list, model catalog, `class_order`, `reference_order`. **503** if no bundle. |
| POST | `/api/predict` | Body: `home_team`, `away_team`, `date` (ISO `YYYY-MM-DD`), `year`, possession/shots/corners/fouls (home & away), `model` (`rf` \| `log` \| `xgb`). Response: `prediction`, `prediction_display`, `probabilities`, **`class_order`** (encoder order for charts). |

Full field validation and examples: **http://127.0.0.1:8000/docs** (when the server is running).

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `EPL_ARTIFACT_PATH` | Absolute path to `bundle.joblib`. Used by `train_model.py` (default `--out`), `ml_core.default_artifact_path()`, and the API `BundleStore`. |
| `EPL_CORS_ORIGINS` | Comma-separated allowed browser origins. If unset: `http://127.0.0.1:8000`, `http://localhost:8000`, and the same hosts on port **8501** (Streamlit). |
| `EPL_CORS_WILDCARD` | Set to `1` or `true` to allow **`*`** (convenient for quick demos; avoid for public production APIs). |
| `EPL_ENV` | `development` (default) or `production` (reserved for future stricter defaults). |

---

## Static UI and the API

1. The browser loads CSS/JS from **`/static/...`**.  
2. **`GET /api/meta`** fills the home/away team `<select>` elements.  
3. **`POST /api/predict`** returns `probabilities` keyed by class string; **`class_order`** defines the order of probability bars in **`web/app.js`** (no hard-coded class order in the client).

---

## Troubleshooting

| Symptom | What to check |
|---------|----------------|
| **`ModuleNotFoundError: app`** | Run Uvicorn and Streamlit from the **project root** (the folder that contains the `app` package), not from inside `app/`. |
| **`503` on `/api/meta` / empty Streamlit** | Run **`python scripts/train_model.py`** and confirm **`artifacts/bundle.joblib`** exists (or set `EPL_ARTIFACT_PATH` to your file). |
| **CSV not found** when training | Ensure **`Football_Dataset_2015_2025.csv`** is in the root, or pass **`--csv "full\path\to\file.csv"`** (quotes help if the path contains spaces). |
| **CORS errors** in the browser | If the UI is on another origin, set **`EPL_CORS_ORIGINS`** or, only for demos, **`EPL_CORS_WILDCARD=1`**. |
| **`LogisticRegression` / `multi_class` TypeError** in notebooks | On scikit-learn **1.6+**, remove `multi_class="multinomial"` from the notebook cell; `ml_core.py` already omits it. |

---

## Git

```powershell
git init
git add .
git commit -m "Add EPL notebooks, ML core, and predictor apps"
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main
```

`.gitignore` excludes `venv/`, `.ipynb_checkpoints/`, **`*.csv`**, and **`artifacts/*`** except **`artifacts/.gitkeep`**.

---

## Keeping notebooks and code in sync

- **`app/ml_core.py`** should mirror the notebook pipeline (EPL filter, feature columns, preprocessors, hyperparameters). If you change the notebook, update `ml_core.py` and **retrain** so the web apps stay consistent.

---

## Data and license

Do not publish the CSV unless its license and your institution or employer rules allow it. This repository does **not** ship the dataset.

If you add a **`LICENSE`** for your own code, keep it separate from third-party data rights.
