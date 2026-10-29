"""FastAPI entry point for the Website Fingerprinting inference service."""

import os
import sys

# Ensure root path is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router, app_state
from src.utils.helpers import load_config, get_logger
from src.models.architectures import build_model

logger = get_logger("AppMain")

app = FastAPI(
    title="Website Fingerprinting Service (Var-CNN & DF)",
    description="REST API for classifying encrypted Tor/network traffic traces into website identities.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def startup_event():
    """Initializes and loads the model on service startup."""
    logger.info("Initializing API inference service...")
    try:
        config = load_config()
        app_state["config"] = config

        model = build_model(config, mixture_num=0)
        app_state["model"] = model

        # Attempt to load trained weights
        artifacts_dir = config.get("artifacts_dir", "artifacts/")
        checkpoint_dir = config.get("checkpoint_dir", os.path.join(artifacts_dir, "checkpoints"))

        weight_candidates = [
            os.path.join(artifacts_dir, "best_model.h5"),
            os.path.join(artifacts_dir, "best_model.pt"),
            os.path.join(checkpoint_dir, "dir_metadata_best_weights.h5"),
            os.path.join(checkpoint_dir, "var_cnn_best_weights.h5"),
            "model_weights.h5"
        ]

        loaded = False
        for weight_path in weight_candidates:
            if os.path.exists(weight_path):
                logger.info(f"Loading model weights from {weight_path}")
                model.load_weights(weight_path)
                loaded = True
                break

        app_state["model_loaded"] = loaded
        if not loaded:
            logger.warning("No pre-trained weights file was found. The model is initialized with random weights.")

        logger.info("Service initialized successfully.")
    except Exception as e:
        logger.error(f"Error during startup model loading: {e}")


@app.get("/")
def root():
    return {
        "service": "Website Fingerprinting API",
        "docs_url": "/docs",
        "health_url": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
