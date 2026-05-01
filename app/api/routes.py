from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.config import get_settings
from app.constants import MODEL_CATALOG
from app.labels import outcome_display
from app.ml_core import ORDER, predict_row
from app.schemas import HealthOut, MetaOut, ModelInfo, PredictIn, PredictOut
from app.services.bundle import BundleStore

router = APIRouter(prefix="/api", tags=["epl"])


def _store(request: Request) -> BundleStore:
    """Lifespan sets bundle_store; lazy-init covers TestClient without context manager."""
    st = request.app.state
    if not hasattr(st, "bundle_store"):
        st.bundle_store = BundleStore(get_settings().artifact_path)
    return st.bundle_store


@router.get("/health", response_model=HealthOut)
def health(request: Request) -> HealthOut:
    store = _store(request)
    b = store.get()
    return HealthOut(
        ok=True,
        model_loaded=b is not None,
        artifact=str(store.path),
    )


@router.get("/meta", response_model=MetaOut)
def meta(request: Request) -> MetaOut:
    b = _store(request).get()
    if b is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model. Run: python scripts/train_model.py (CSV required).",
        )
    le = b["label_encoder"]
    classes = [str(c) for c in le.classes_.tolist()]
    return MetaOut(
        teams=b["teams"],
        models=[ModelInfo(**m) for m in MODEL_CATALOG],
        class_order=classes,
        reference_order=list(ORDER),
    )


@router.post("/predict", response_model=PredictOut)
def predict(request: Request, body: PredictIn) -> PredictOut:
    b = _store(request).get()
    if b is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model. Run: python scripts/train_model.py",
        )
    if body.model not in b["models"]:
        raise HTTPException(400, "model must be rf, log, or xgb")

    row = body.model_dump()
    model_key = row.pop("model")
    try:
        raw, proba = predict_row(b, model_key, row)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:
        raise HTTPException(400, str(e)) from e

    le = b["label_encoder"]
    class_order = [str(x) for x in le.classes_.tolist()]
    probs = {class_order[i]: float(proba[i]) for i in range(len(class_order))}

    return PredictOut(
        prediction=raw,
        prediction_display=outcome_display(raw),
        probabilities=probs,
        class_order=class_order,
        model=model_key,
    )


@router.get("/")
def api_root():
    return {"message": "EPL predictor API", "docs": "/docs"}
