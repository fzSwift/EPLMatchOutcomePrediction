"""
Streamlit UI for EPL match outcome prediction (same models as app/ml_core.py).

Run from project root:
  streamlit run streamlit_app.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.constants import MODEL_CATALOG
from app.labels import outcome_display
from app.ml_core import default_artifact_path, load_bundle, predict_row, train_bundle

MODEL_OPTIONS = {m["name"]: m["id"] for m in MODEL_CATALOG}
DEFAULT_CSV_NAME = "Football_Dataset_2015_2025.csv"


@st.cache_resource
def bundle_for_mtime(mtime: float):
    """Reload when bundle.joblib is replaced (mtime passed as cache key)."""
    return load_bundle()


def _default_csv_path() -> Path:
    env = os.environ.get("EPL_DATASET_PATH")
    if env:
        return Path(env)
    return ROOT / DEFAULT_CSV_NAME


def _train_and_save_bundle(csv_path: Path, artifact_path: Path):
    with st.spinner(f"Training models from `{csv_path.name}`..."):
        bundle = train_bundle(csv_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)
    return bundle


def _load_or_train_bundle(artifact_path: Path):
    if artifact_path.is_file():
        return bundle_for_mtime(artifact_path.stat().st_mtime)

    st.warning(f"No model found at `{artifact_path}`.")
    csv_path = _default_csv_path()
    if csv_path.is_file():
        st.info(f"Found dataset at `{csv_path}`. Training automatically now.")
        _train_and_save_bundle(csv_path, artifact_path)
        return bundle_for_mtime(artifact_path.stat().st_mtime)

    st.error(
        "Model artifact and dataset are both missing. "
        f"Add `{DEFAULT_CSV_NAME}` to the project root, set `EPL_DATASET_PATH`, "
        "or upload a CSV below."
    )
    upload = st.file_uploader("Upload dataset CSV", type=["csv"])
    if upload is None:
        st.stop()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        tf.write(upload.getvalue())
        temp_path = Path(tf.name)
    try:
        _train_and_save_bundle(temp_path, artifact_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return bundle_for_mtime(artifact_path.stat().st_mtime)


def main():
    st.set_page_config(page_title="EPL predictor", layout="wide")
    st.title("EPL match outcome predictor")
    st.caption(
        "Uses the same training/prediction pipeline as notebooks and FastAPI. "
        "If the bundle is missing, this page can train it automatically."
    )

    artifact_path = default_artifact_path()
    try:
        bundle = _load_or_train_bundle(artifact_path)
    except Exception as e:
        st.error(f"Failed to initialize model bundle: {e}")
        st.stop()

    teams: list[str] = bundle["teams"]
    le = bundle["label_encoder"]
    class_labels = [str(c) for c in le.classes_.tolist()]

    with st.sidebar:
        st.subheader("Model")
        model_name = st.radio("Classifier", list(MODEL_OPTIONS.keys()), index=0)
        model_key = MODEL_OPTIONS[model_name]
        st.divider()
        st.subheader("Artifact")
        st.code(str(artifact_path))
        st.subheader("Test metrics (hold-out)")
        for k, v in bundle.get("metrics", {}).items():
            st.markdown(f"**{k}**")
            if isinstance(v, dict):
                st.json(v)

    c1, c2 = st.columns(2)
    with c1:
        match_date = st.date_input("Match date", value=date.today())
    with c2:
        year = st.number_input("Season year", min_value=2015, max_value=2035, value=2024)

    left, right = st.columns(2)
    with left:
        st.markdown("### Home")
        home_team = st.selectbox("Home team", teams, key="h")
        ph = st.slider("Possession % (home)", 0, 100, 55)
        sh = st.slider("Shots (home)", 0, 40, 14)
        ch = st.slider("Corners (home)", 0, 20, 6)
        fh = st.slider("Fouls (home)", 0, 25, 12)
    with right:
        st.markdown("### Away")
        away_idx = 1 if len(teams) > 1 else 0
        away_team = st.selectbox(
            "Away team",
            teams,
            index=min(away_idx, len(teams) - 1),
            key="a",
        )
        pa = st.slider("Possession % (away)", 0, 100, 45)
        sa = st.slider("Shots (away)", 0, 40, 10)
        ca = st.slider("Corners (away)", 0, 20, 4)
        fa = st.slider("Fouls (away)", 0, 25, 11)

    if home_team == away_team:
        st.warning("Home and away teams should differ for a meaningful prediction.")

    if st.button("Predict", type="primary", use_container_width=True):
        row = {
            "date": match_date.isoformat(),
            "year": int(year),
            "home_team": home_team,
            "away_team": away_team,
            "possession_home": int(ph),
            "possession_away": int(pa),
            "shots_home": int(sh),
            "shots_away": int(sa),
            "corners_home": int(ch),
            "corners_away": int(ca),
            "fouls_home": int(fh),
            "fouls_away": int(fa),
        }
        try:
            raw, proba = predict_row(bundle, model_key, row)
        except Exception as e:
            st.error(str(e))
            st.stop()

        probs = {class_labels[i]: float(proba[i]) for i in range(len(class_labels))}
        st.success(f"**{outcome_display(raw)}** (`{raw}`)")

        chart_df = pd.DataFrame(
            {
                "Outcome": [outcome_display(k) for k in class_labels],
                "Probability": [probs[k] for k in class_labels],
            }
        ).set_index("Outcome")
        st.bar_chart(chart_df, height=260)


if __name__ == "__main__":
    main()
