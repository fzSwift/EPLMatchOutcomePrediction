"""
Shared ML logic aligned with EPL_Project_Notebook.ipynb:
filter to Premier League, features, preprocessor, three classifiers.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

RANDOM_STATE = 42
ORDER = ["Home Team", "Draw", "Away Team"]

FEATURE_COLS = [
    "Home Team",
    "Away Team",
    "Year",
    "Possession % (Home)",
    "Possession % (Away)",
    "Shots (Home)",
    "Shots (Away)",
    "Corners (Home)",
    "Corners (Away)",
    "Fouls (Home)",
    "Fouls (Away)",
    "Month",
    "DayOfWeek",
]
CAT_COLS = ["Home Team", "Away Team"]
NUM_COLS = [c for c in FEATURE_COLS if c not in CAT_COLS]


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def build_epl_subset(df: pd.DataFrame) -> pd.DataFrame:
    epl = df[df["Competition"] == "Premier League"].copy()
    epl = epl.dropna(subset=["Winner", "Date"])
    epl["Year"] = pd.to_numeric(epl["Year"], errors="coerce")
    epl["Month"] = epl["Date"].dt.month
    epl["DayOfWeek"] = epl["Date"].dt.dayofweek
    return epl


def xy_from_epl(epl: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = epl[FEATURE_COLS].copy()
    y = epl["Winner"].copy()
    for c in NUM_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, y


def make_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        [("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_transformer, NUM_COLS),
            ("cat", categorical_transformer, CAT_COLS),
        ]
    )


def build_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    log_reg = LogisticRegression(
        max_iter=5000,
        random_state=RANDOM_STATE,
    )
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=3,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
    )
    return {
        "rf": Pipeline([("preprocess", clone(preprocessor)), ("clf", rf)]),
        "log": Pipeline([("preprocess", clone(preprocessor)), ("clf", log_reg)]),
        "xgb": Pipeline([("preprocess", clone(preprocessor)), ("clf", xgb)]),
    }


def train_bundle(csv_path: str | Path) -> dict[str, Any]:
    """Train all models; return serializable bundle for joblib."""
    df = load_raw_csv(csv_path)
    epl = build_epl_subset(df)
    X, y = xy_from_epl(epl)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pre = make_preprocessor()
    models = build_models(pre)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    for name, pipe in models.items():
        pipe.fit(X_train, y_train_enc)

    metrics = {}
    for name, m in models.items():
        yp_enc = m.predict(X_test)
        yp = le.inverse_transform(yp_enc)
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, yp)),
            "macro_f1": float(f1_score(y_test, yp, average="macro")),
            "weighted_f1": float(f1_score(y_test, yp, average="weighted")),
        }

    teams = sorted(
        pd.unique(pd.concat([epl["Home Team"], epl["Away Team"]], ignore_index=True))
        .tolist()
    )

    return {
        "models": models,
        "label_encoder": le,
        "metrics": metrics,
        "teams": teams,
        "feature_cols": FEATURE_COLS,
        "order": ORDER,
    }


def predict_row(
    bundle: dict[str, Any],
    model_key: str,
    row: dict[str, Any],
) -> tuple[str, np.ndarray]:
    """row keys match API / form: snake_case mapped internally."""
    m = bundle["models"][model_key]
    le: LabelEncoder = bundle["label_encoder"]

    dt = pd.to_datetime(row["date"], errors="coerce")
    if pd.isna(dt):
        raise ValueError("Invalid or missing date; use ISO format YYYY-MM-DD.")

    frame = pd.DataFrame(
        [
            {
                "Home Team": row["home_team"],
                "Away Team": row["away_team"],
                "Year": int(row["year"]),
                "Possession % (Home)": int(row["possession_home"]),
                "Possession % (Away)": int(row["possession_away"]),
                "Shots (Home)": int(row["shots_home"]),
                "Shots (Away)": int(row["shots_away"]),
                "Corners (Home)": int(row["corners_home"]),
                "Corners (Away)": int(row["corners_away"]),
                "Fouls (Home)": int(row["fouls_home"]),
                "Fouls (Away)": int(row["fouls_away"]),
                "Month": int(dt.month),
                "DayOfWeek": int(dt.dayofweek),
            }
        ]
    )

    enc = m.predict(frame)
    label = le.inverse_transform(enc)[0]
    proba = m.predict_proba(frame)[0]
    return str(label), proba


def default_artifact_path() -> Path:
    env = os.environ.get("EPL_ARTIFACT_PATH")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "artifacts" / "bundle.joblib"


def load_bundle(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else default_artifact_path()
    return joblib.load(p)
