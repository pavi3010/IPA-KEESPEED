"""
OpenAI-compatible FastAPI server for DeepSeek.

Point any OpenAI client at either:
    - http://localhost:8000/v1
    - http://localhost:8000

Example:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Hello!"}],
    )

Endpoints supported (both with and without /v1 prefix):
    POST /v1/chat/completions or /chat/completions (stream=true supported)
    GET  /v1/models or /models
    GET  /v1/models/{model} or /models/{model}
    POST /v1/embeddings or /embeddings
    GET  /healthz or /
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from deepseek.auth import LoginRequired
from deepseek.client import DeepSeekClient, UpstreamError

from .config import (
    API_KEY,
    MODEL_MAP,
    RATE_LIMIT_PER_MINUTE,
    SERVER_INTERACTIVE_LOGIN,
    is_known_model,
    resolve_model_type,
    should_enable_thinking,
)
from .openai_format import (
    completion_response,
    extract_tool_calls,
    messages_to_prompt,
    stream_chunks,
    stream_chunks_with_tools,
    trim_at_role_boundary,
)
from .ratelimit import RateLimiter, install_rate_limit
from .schemas import ChatCompletionRequest, EmbeddingRequest

load_dotenv()

# uvicorn configures logging with disable_existing_loggers, so hang our
# debug output off its own logger to be sure it reaches the console.
log = logging.getLogger("uvicorn.error")

# Set DEBUG_REQUESTS=1 to log each request's shape (roles + tool names). Useful
# when a client's tool calls aren't arriving.
DEBUG_REQUESTS = os.getenv("DEBUG_REQUESTS", "").lower() in ("1", "true", "yes", "on")

# Seconds to advertise in Retry-After when DeepSeek rate-limits us.
UPSTREAM_RETRY_AFTER = os.getenv("UPSTREAM_RETRY_AFTER", "30")

app = FastAPI(title="DeepSeek OpenAI-compatible API", version="0.2.0")

# Enable CORS for browser frontends (NextChat, OpenWebUI, LibreChat, n8n, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

install_rate_limit(app, RateLimiter(limit=RATE_LIMIT_PER_MINUTE, window=60.0))

# One shared client (and its signed-in session) built lazily on first use.
_client: DeepSeekClient | None = None
_client_lock = threading.Lock()


def get_client() -> DeepSeekClient:
    """Build (once) the shared client and its signed-in session."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = DeepSeekClient(allow_interactive=SERVER_INTERACTIVE_LOGIN)
    return _client


def _error(message: str, status: int = 500, err_type: str = "server_error",
           headers: dict = None, code: Optional[str] = None):
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": err_type,
                "param": None,
                "code": code or str(status),
            }
        },
        headers=headers or None,
    )


def _upstream_error(exc: UpstreamError):
    """Map a DeepSeek stream error onto the status a client can act on."""
    if exc.is_rate_limit:
        return _error(str(exc), status=429, err_type="rate_limit_exceeded",
                      headers={"Retry-After": UPSTREAM_RETRY_AFTER})
    return _error(f"DeepSeek reported: {exc}", status=502, err_type="upstream_error")


def _verify_auth(authorization: Optional[str] = None):
    """If API_KEY is configured in the environment, verify incoming Bearer token."""
    if not API_KEY:
        return True
    if not authorization:
        return False
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1] == API_KEY
    return authorization == API_KEY


class _Prefetched:
    """A stream with its first delta already pulled.

    The endpoint consumes one delta before returning a response, so an error
    frame -- which DeepSeek sends immediately -- becomes a real status code
    instead of a 200 whose body turns out to be an error mid-flight.
    """

    def __init__(self, stream, iterator, first):
        self._stream, self._iter, self._first = stream, iterator, first

    def __iter__(self):
        if self._first is not None:
            yield self._first
        yield from self._iter

    @property
    def conversation_id(self):
        return getattr(self._stream, "conversation_id", None)


@app.get("/")
@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": "deepseek-openai-api"}


@app.get("/models")
@app.get("/v1/models")
def list_models(authorization: Optional[str] = Header(None)):
    if not _verify_auth(authorization):
        return _error("Incorrect API key provided.", status=401, err_type="invalid_api_key")

    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": created,
                "owned_by": "deepseek",
                "permission": [],
                "root": name,
                "parent": None,
            }
            for name in MODEL_MAP
        ],
    }


@app.get("/models/{model}")
@app.get("/v1/models/{model}")
def retrieve_model(model: str, authorization: Optional[str] = Header(None)):
    if not _verify_auth(authorization):
        return _error("Incorrect API key provided.", status=401, err_type="invalid_api_key")

    if not is_known_model(model):
        return _error(f"The model `{model}` does not exist", status=404, err_type="model_not_found")

    return {
        "id": model,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "deepseek",
        "permission": [],
        "root": model,
        "parent": None,
    }


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
):
    if not _verify_auth(authorization):
        return _error("Incorrect API key provided.", status=401, err_type="invalid_api_key")

    if not req.messages:
        return _error("`messages` must not be empty", status=400, err_type="invalid_request_error")

    # A thread's model is fixed when it's created, so on resume we ignore `model`
    # and let the existing thread's model stand.
    model_type = None if req.conversation_id else resolve_model_type(req.model)
    thinking_enabled = should_enable_thinking(req.model, req.thinking)
    search_enabled = bool(req.search)

    if DEBUG_REQUESTS:
        log.warning("request: model=%s stream=%s roles=%s tools=%s tool_choice=%s",
                    req.model, req.stream, [m.role for m in req.messages],
                    [ (t.get("function") or t).get("name") for t in (req.tools or []) ],
                    req.tool_choice)

    prompt = messages_to_prompt(req.messages, req.tools, req.tool_choice)
    tools = None if req.tool_choice == "none" else req.tools

    try:
        client = await run_in_threadpool(get_client)
    except LoginRequired as e:
        return _error(str(e), status=503, err_type="login_required")
    except Exception as e:
        return _error(f"Failed to initialise DeepSeek session: {e}")

    include_usage = bool(req.stream_options and req.stream_options.include_usage)

    if req.stream:
        raw = client.stream(
            prompt, conversation_id=req.conversation_id,
            model=model_type, thinking=thinking_enabled, search=search_enabled,
        )
        it = iter(raw)
        try:
            first = await run_in_threadpool(lambda: next(it, None))
        except UpstreamError as e:
            return _upstream_error(e)
        except Exception as e:
            return _error(f"DeepSeek request failed: {e}")

        def gen():
            stream = _Prefetched(raw, it, first)
            if tools:
                yield from stream_chunks_with_tools(
                    req.model, stream, tools, prompt=prompt, include_usage=include_usage
                )
            else:
                yield from stream_chunks(
                    req.model, stream, prompt=prompt, include_usage=include_usage
                )

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            },
        )

    try:
        reply = await run_in_threadpool(
            client.chat, prompt, req.conversation_id,
            model_type, thinking_enabled, search_enabled,
        )
    except UpstreamError as e:
        return _upstream_error(e)
    except Exception as e:
        return _error(f"DeepSeek request failed: {e}")

    calls, text = extract_tool_calls(reply.text, tools)
    if not tools:
        text = trim_at_role_boundary(text)
    if DEBUG_REQUESTS:
        log.warning("reply: calls=%s text=%r",
                    [(c["function"]["name"], c["function"]["arguments"]) for c in calls or []],
                    (text or "")[:200])

    return completion_response(
        req.model, text, prompt, reply.conversation_id, tool_calls=calls
    )


@app.post("/embeddings")
@app.post("/v1/embeddings")
async def embeddings(
    req: EmbeddingRequest,
    authorization: Optional[str] = Header(None),
):
    """Fallback embeddings endpoint so OpenAI clients / agents checking embeddings do not break."""
    if not _verify_auth(authorization):
        return _error("Incorrect API key provided.", status=401, err_type="invalid_api_key")

    inputs = [req.input] if isinstance(req.input, str) else req.input
    data = []
    for idx, _ in enumerate(inputs):
        data.append({
            "object": "embedding",
            "index": idx,
            "embedding": [0.0] * 1536,
        })

    return {
        "object": "list",
        "data": data,
        "model": req.model or "text-embedding-ada-002",
        "usage": {
            "prompt_tokens": len(inputs) * 5,
            "total_tokens": len(inputs) * 5,
        },
    }
