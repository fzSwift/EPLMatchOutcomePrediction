from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.config import get_settings
from app.services.bundle import BundleStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.bundle_store = BundleStore(settings.artifact_path)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="EPL outcome predictor",
        description="API for the EPL notebooks — train with scripts/train_model.py first.",
        lifespan=lifespan,
    )

    # credentials=False so allow_origins=["*"] remains valid when EPL_CORS_WILDCARD=1
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    if settings.web_dir.is_dir():
        app.mount(
            "/static",
            StaticFiles(directory=str(settings.web_dir)),
            name="static",
        )

        @app.get("/")
        def serve_ui():
            index = settings.web_dir / "index.html"
            if not index.is_file():
                raise HTTPException(404, "web/index.html missing")
            return FileResponse(index)

    return app


app = create_app()
