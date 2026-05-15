"""
FastAPI application entry point for the SHL Assessment Recommender.

Endpoints:
  GET  /health  →  {"status": "ok"}   (liveness probe, up to 2 min cold start)
  POST /chat    →  ChatResponse        (stateless, full history per call)

The catalog index is built once at startup via the lifespan context manager.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.catalog import catalog
from app.models import ChatRequest, ChatResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — build catalog index on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== SHL Recommender starting up ===")
    catalog.load()
    logger.info("=== Catalog ready — service is live ===")
    yield
    logger.info("=== SHL Recommender shutting down ===")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "Conversational agent that recommends SHL Individual Test assessments "
        "based on a multi-turn dialogue with hiring managers and recruiters."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception handler — ensures we never return non-JSON on errors
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
async def health():
    """
    Liveness probe.
    Returns HTTP 200 with {"status": "ok"} once the catalog is loaded.
    The evaluator allows up to 2 minutes on cold start.
    """
    if not catalog.items:
        raise HTTPException(status_code=503, detail="Catalog not yet loaded")
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Stateless conversational endpoint.
    The caller sends the full message history on every turn.
    Returns the agent's reply plus an optional structured shortlist.
    """
    from app.agent import run_agent  # lazy import so lifespan starts first

    if not request.messages:
        raise HTTPException(status_code=422, detail="messages list cannot be empty")

    # Quick guard: last message must be from the user
    if request.messages[-1].role != "user":
        raise HTTPException(
            status_code=422,
            detail="The last message in the history must have role='user'",
        )

    response = await run_agent(request)
    return response
