"""
AILS FastAPI REST Service
Production-ready REST API for AILS model inference.
Artificial Intelligence Learning System — Created by Cherry Computer Ltd.
"""

import logging
import numpy as np
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logger = logging.getLogger("AILS.API")


# ── Request / Response Schemas ────────────────────────────────────────────────

class TextRequest(BaseModel):
    """Request body for single-text inference."""
    text: str = Field(..., min_length=1, max_length=10000,
                      description="Input text for analysis")
    include_score: bool = Field(default=True,
                                description="Return confidence score")


class BatchTextRequest(BaseModel):
    """Request body for batch text inference."""
    texts: List[str] = Field(..., min_items=1, max_items=100)


class SentimentResponse(BaseModel):
    """Sentiment analysis response."""
    text: str
    sentiment: str
    confidence: float
    positive_score: float
    negative_score: float


class HealthResponse(BaseModel):
    """API health-check response."""
    status: str
    version: str
    creator: str
    model_loaded: bool


class FairnessRequest(BaseModel):
    """Request body for fairness audit."""
    y_true: List[int]
    y_pred: List[int]
    sensitive_attr: List[int]
    privileged_group: int = 1


# ── App Factory ───────────────────────────────────────────────────────────────

def create_app() -> "FastAPI":
    """
    Create and configure the AILS FastAPI application.

    Returns:
        Configured FastAPI app instance.
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

    app = FastAPI(
        title="AILS REST API",
        description=(
            "**Artificial Intelligence Learning System** REST API\n\n"
            "Provides sentiment analysis, NLP inference, and AI fairness "
            "auditing endpoints.\n\n"
            "*Created and maintained by **Cherry Computer Ltd.***"
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        contact={
            "name": "Cherry Computer Ltd.",
            "email": "contact@cherrycomputer.ltd",
            "url": "https://github.com/CherryComputerLtd/Artificial-Intelligence-Learning-System-AILS-",
        },
        license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Lazy-loaded components ────────────────────────────────────────────────
    _state: Dict[str, Any] = {"sentiment_analyzer": None, "bias_detector": None}

    @app.on_event("startup")
    async def startup():
        """Initialize AILS components on startup."""
        try:
            from src.nlp.sentiment import SentimentAnalyzer
            from src.ethics.bias_detector import AILSBiasDetector
            _state["sentiment_analyzer"] = SentimentAnalyzer()
            _state["bias_detector"] = AILSBiasDetector()
            logger.info("✅ AILS API components initialized.")
        except Exception as e:
            logger.error(f"⚠️  Startup warning: {e}")

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/", tags=["Info"])
    async def root():
        """API root — returns project information."""
        return {
            "name": "AILS REST API",
            "full_name": "Artificial Intelligence Learning System",
            "version": "1.0.0",
            "creator": "Cherry Computer Ltd.",
            "repository": (
                "https://github.com/CherryComputerLtd/"
                "Artificial-Intelligence-Learning-System-AILS-"
            ),
            "docs": "/docs",
        }

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health():
        """API health check endpoint."""
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            creator="Cherry Computer Ltd.",
            model_loaded=_state["sentiment_analyzer"] is not None,
        )

    @app.post("/analyze/sentiment",
              response_model=SentimentResponse,
              tags=["NLP"])
    async def analyze_sentiment(request: TextRequest):
        """
        Analyze the sentiment of input text.

        Returns positive, negative, or neutral with confidence scores.
        """
        analyzer = _state.get("sentiment_analyzer")
        if analyzer is None:
            raise HTTPException(status_code=503, detail="Analyzer not loaded")

        results = analyzer.analyze_with_scores([request.text])
        r = results[0]
        return SentimentResponse(
            text=request.text[:200],
            sentiment=r["sentiment"],
            confidence=r["confidence"],
            positive_score=r["positive_score"],
            negative_score=r["negative_score"],
        )

    @app.post("/analyze/sentiment/batch", tags=["NLP"])
    async def analyze_sentiment_batch(request: BatchTextRequest):
        """Batch sentiment analysis — analyze up to 100 texts at once."""
        analyzer = _state.get("sentiment_analyzer")
        if analyzer is None:
            raise HTTPException(status_code=503, detail="Analyzer not loaded")

        results = analyzer.analyze_with_scores(request.texts)
        return {"count": len(results), "results": results}

    @app.post("/ethics/fairness", tags=["Ethics"])
    async def fairness_audit(request: FairnessRequest):
        """
        Run a full AI fairness audit.

        Computes demographic parity, equalized odds, and disparate impact.
        """
        detector = _state.get("bias_detector")
        if detector is None:
            raise HTTPException(status_code=503, detail="Bias detector not loaded")

        y_true = np.array(request.y_true)
        y_pred = np.array(request.y_pred)
        sensitive = np.array(request.sensitive_attr)

        report = detector.generate_fairness_report(
            y_true, y_pred, sensitive, request.privileged_group
        )
        return {"status": "ok", "report": report}

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(exc)},
        )

    return app


# ── Entry Point ───────────────────────────────────────────────────────────────
app = create_app() if FASTAPI_AVAILABLE else None


def run_server(host: str = "0.0.0.0", port: int = 8000,
               reload: bool = False) -> None:
    """Start the AILS API server."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI/uvicorn required: pip install fastapi uvicorn")
    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
