"""
main.py — FastAPI service exposing:
  GET  /health   → {"status": "ok"}
  POST /chat     → {reply, recommendations, end_of_conversation}

Startup:
    uvicorn main:app --host 0.0.0.0 --port 8000

Environment variables:
    ANTHROPIC_API_KEY   (required)
    CATALOG_PATH        (optional, default: catalog.json)
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from agent import SHLAgent
from retrieval import CatalogRetriever

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models (exact schema required by the evaluator)
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)

    @field_validator("messages")
    @classmethod
    def last_must_be_user(cls, msgs: list[Message]) -> list[Message]:
        if msgs and msgs[-1].role != "user":
            raise ValueError("The last message must have role='user'.")
        return msgs


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# ---------------------------------------------------------------------------
# App & lifespan
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for recommending SHL Individual Test Solutions.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Singleton instances — built once at startup
_retriever: CatalogRetriever | None = None
_agent: SHLAgent | None = None
_startup_error: str | None = None


@app.on_event("startup")
async def _startup() -> None:
    global _retriever, _agent, _startup_error
    try:
        catalog_path = Path(os.environ.get("CATALOG_PATH", "catalog.json"))
        log.info("Loading catalog from %s …", catalog_path)
        _retriever = CatalogRetriever(catalog_path)
        _agent = SHLAgent(_retriever)
        log.info("Agent ready.")
    except Exception as exc:
        _startup_error = str(exc)
        log.error("Startup failed: %s", exc)


# ---------------------------------------------------------------------------
# Middleware: request timing
# ---------------------------------------------------------------------------

@app.middleware("http")
async def _add_timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - t0
    response.headers["X-Response-Time-Ms"] = f"{elapsed * 1000:.1f}"
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=dict)
async def health() -> dict:
    """Readiness probe. Returns 200 when the agent is ready, 503 otherwise."""
    if _startup_error:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": _startup_error},
        )
    if _agent is None:
        return JSONResponse(status_code=503, content={"status": "initialising"})
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Stateless chat endpoint.

    The caller supplies the full conversation history on every request.
    The agent processes it and returns the next reply plus optional recommendations.
    """
    if _agent is None:
        if _startup_error:
            raise HTTPException(status_code=503, detail=f"Service unavailable: {_startup_error}")
        raise HTTPException(status_code=503, detail="Service is still initialising.")

    # Convert Pydantic models to plain dicts for the agent
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Cap at 8 turns (user + assistant)
    if len(messages) > 8:
        messages = messages[-8:]

    log.info("chat() — %d messages, last user: %r", len(messages), messages[-1]["content"][:80])

    result: dict[str, Any] = _agent.chat(messages)

    return ChatResponse(
        reply=result["reply"],
        recommendations=[
            Recommendation(
                name=r["name"],
                url=r["url"],
                test_type=r["test_type"],
            )
            for r in result.get("recommendations", [])
        ],
        end_of_conversation=result.get("end_of_conversation", False),
    )


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
