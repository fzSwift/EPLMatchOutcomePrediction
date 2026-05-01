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
from urllib.request import urlretrieve

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.constants import MODEL_CATALOG
from app.labels import outcome_display
from app.ml_core import default_artifact_path, load_bundle, predict_row, train_bundle

MODEL_OPTIONS = {m["name"]: m["id"] for m in MODEL_CATALOG}
DEFAULT_CSV_NAME = "Football_Dataset_2015_2025.csv"
OUTCOME_COLORS = {"Home Team": "#2196F3", "Draw": "#9E9E9E", "Away Team": "#F44336"}


@st.cache_resource
def bundle_for_mtime(mtime: float):
    """Reload when bundle.joblib is replaced (mtime passed as cache key)."""
    return load_bundle()


def _default_csv_path() -> Path:
    env = os.environ.get("EPL_DATASET_PATH")
    if env:
        return Path(env)
    return ROOT / DEFAULT_CSV_NAME


def _download_csv_if_configured() -> Path | None:
    url = os.environ.get("EPL_DATASET_URL", "").strip()
    if not url:
        return None
    target = ROOT / DEFAULT_CSV_NAME
    with st.spinner("Downloading dataset from EPL_DATASET_URL..."):
        urlretrieve(url, target)
    st.success(f"Downloaded dataset to `{target}`.")
    return target


def _train_and_save_bundle(csv_path: Path, artifact_path: Path):
    with st.spinner(f"Training models from `{csv_path.name}`..."):
        bundle = train_bundle(csv_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)
    return bundle


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
          .main h1 { font-size: 2rem; margin-bottom: 0.2rem; }
          .subtext { color: #9ca3af; margin-top: 0; margin-bottom: 1rem; }
          .card {
            background: linear-gradient(180deg, #111827 0%, #0b1220 100%);
            border: 1px solid #1f2937;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.75rem;
          }
          .card h4 { margin: 0 0 0.35rem 0; font-size: 0.95rem; color: #d1d5db; }
          .pred {
            border-left: 4px solid #60a5fa;
            background: #0b1220;
            border-radius: 10px;
            padding: 0.75rem 0.9rem;
            margin: 0.4rem 0 0.8rem;
            font-size: 1.05rem;
          }
          .section-title { margin-top: 0.35rem; margin-bottom: 0.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_probability_bars(class_labels: list[str], probs: dict[str, float]) -> None:
    st.markdown("#### Outcome Probabilities")
    for label in class_labels:
        display = outcome_display(label)
        p = float(probs.get(label, 0.0))
        color = OUTCOME_COLORS.get(label, "#64748b")
        st.markdown(
            (
                f"<div style='display:flex;justify-content:space-between;"
                f"margin:0.2rem 0 0.2rem;'>"
                f"<span>{display}</span><span>{p*100:.1f}%</span></div>"
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            (
                "<div style='background:#1f2937;border-radius:999px;height:10px;'>"
                f"<div style='width:{p*100:.1f}%;background:{color};"
                "height:10px;border-radius:999px;'></div></div>"
            ),
            unsafe_allow_html=True,
        )


def _load_or_train_bundle(artifact_path: Path):
    if artifact_path.is_file():
        return bundle_for_mtime(artifact_path.stat().st_mtime)

    st.warning(f"No model found at `{artifact_path}`.")
    csv_path = _default_csv_path()
    if csv_path.is_file():
        st.info(f"Found dataset at `{csv_path}`. Training automatically now.")
        _train_and_save_bundle(csv_path, artifact_path)
        return bundle_for_mtime(artifact_path.stat().st_mtime)

    downloaded = _download_csv_if_configured()
    if downloaded and downloaded.is_file():
        _train_and_save_bundle(downloaded, artifact_path)
        return bundle_for_mtime(artifact_path.stat().st_mtime)

    st.error(
        "Model artifact and dataset are both missing. "
        f"Add `{DEFAULT_CSV_NAME}` to the project root, set `EPL_DATASET_PATH`, "
        "set `EPL_DATASET_URL`, or upload a CSV below."
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
    _inject_styles()
    st.title("EPL match outcome predictor")
    st.markdown(
        "<p class='subtext'>Uses the same training/prediction pipeline as notebooks "
        "and FastAPI. If the bundle is missing, this page can train it automatically.</p>",
        unsafe_allow_html=True,
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
            st.markdown(f"<div class='card'><h4>{k}</h4></div>", unsafe_allow_html=True)
            if isinstance(v, dict):
                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy", f"{v.get('accuracy', 0)*100:.1f}%")
                c2.metric("Macro F1", f"{v.get('macro_f1', 0):.3f}")
                c3.metric("Weighted F1", f"{v.get('weighted_f1', 0):.3f}")

    top1, top2 = st.columns(2)
    with top1:
        match_date = st.date_input("Match date", value=date.today())
    with top2:
        year = st.number_input("Season year", min_value=2015, max_value=2035, value=2024)

    left, right = st.columns(2)
    with left:
        st.markdown("### Home Team")
        home_team = st.selectbox("Home team", teams, key="h")
        ph = st.slider("Possession % (home)", 0, 100, 55)
        sh = st.slider("Shots (home)", 0, 40, 14)
        ch = st.slider("Corners (home)", 0, 20, 6)
        fh = st.slider("Fouls (home)", 0, 25, 12)
    with right:
        st.markdown("### Away Team")
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
        st.markdown(
            f"<div class='pred'><strong>Prediction:</strong> {outcome_display(raw)} "
            f"(<code>{raw}</code>)</div>",
            unsafe_allow_html=True,
        )

        col_left, col_right = st.columns([3, 2])
        with col_left:
            _render_probability_bars(class_labels, probs)
        with col_right:
            st.markdown("#### Match Summary")
            st.markdown(
                f"""
                <div class='card'>
                <h4>{home_team} vs {away_team}</h4>
                <div>Date: {match_date.isoformat()}</div>
                <div>Season year: {int(year)}</div>
                <div>Model: {model_name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
