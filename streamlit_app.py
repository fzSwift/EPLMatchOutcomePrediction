"""
Streamlit UI for EPL match outcome prediction (same models as app/ml_core.py).

Run from project root:
  streamlit run streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import date

import pandas as pd
import streamlit as st

from app.constants import MODEL_CATALOG
from app.labels import outcome_display
from app.ml_core import default_artifact_path, load_bundle, predict_row

MODEL_OPTIONS = {m["name"]: m["id"] for m in MODEL_CATALOG}


@st.cache_resource
def bundle_for_mtime(mtime: float):
    """Reload when bundle.joblib is replaced (mtime passed as cache key)."""
    return load_bundle()


def main():
    st.set_page_config(page_title="EPL predictor", layout="wide")
    st.title("EPL match outcome predictor")
    st.caption(
        "Same pipelines as the notebooks / FastAPI app. "
        "Train first: `python scripts/train_model.py`"
    )

    path = default_artifact_path()
    if not path.is_file():
        st.error(
            f"No model at `{path}`. Run **`python scripts/train_model.py`** "
            "with `Football_Dataset_2015_2025.csv` in the project folder."
        )
        st.stop()

    mtime = path.stat().st_mtime
    try:
        bundle = bundle_for_mtime(mtime)
    except Exception as e:
        st.error(f"Failed to load model bundle: {e}")
        st.stop()

    teams: list[str] = bundle["teams"]
    le = bundle["label_encoder"]
    class_labels = [str(c) for c in le.classes_.tolist()]

    with st.sidebar:
        st.subheader("Model")
        model_name = st.radio("Classifier", list(MODEL_OPTIONS.keys()), index=0)
        model_key = MODEL_OPTIONS[model_name]
        st.divider()
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
                "Outcome": [
                    outcome_display(k) for k in class_labels
                ],
                "Probability": [probs[k] for k in class_labels],
            }
        ).set_index("Outcome")
        st.bar_chart(chart_df, height=260)


if __name__ == "__main__":
    main()
