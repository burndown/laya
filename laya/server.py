"""Small HTTP adapter for Laya's typed-decision Router."""

import argparse
import hmac
import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Callable, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse


MODEL_IDS = {
    "laya-auto": None,
    "laya-english": "english",
    "laya-multilingual": "multilingual",
    "laya-typed-decisions": "typed-decisions",
}


def _bad(message: str) -> None:
    raise HTTPException(status_code=400, detail=message)


def _question_payload(payload: Any) -> tuple[Any, dict, str]:
    if not isinstance(payload, dict):
        _bad("Request body must be a JSON object")
    if set(payload) - {"state", "questions", "model"}:
        _bad("Unsupported predict fields: " + ", ".join(sorted(set(payload) - {"state", "questions", "model"})))
    state = payload.get("state")
    if not isinstance(state, (str, dict, list)):
        _bad("state must be a string, object, or array")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        _bad("questions must be a nonempty object")
    for qid, question in questions.items():
        if not isinstance(qid, str) or not qid or not isinstance(question, dict):
            _bad("Each question needs a nonempty ID and an object definition")
        if set(question) - {"type", "instructions", "criteria"}:
            _bad("Unsupported fields in question " + qid)
        kind = question.get("type")
        if kind not in ("choice", "score", "noul"):
            _bad("Question " + qid + " has unsupported type")
        if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
            _bad("Question " + qid + " needs nonempty instructions")
        criteria = question.get("criteria")
        if kind == "noul":
            if "criteria" in question:
                _bad("noul question " + qid + " must not have criteria")
        elif kind == "choice":
            if isinstance(criteria, dict):
                valid = len(criteria) >= 2 and all(
                    isinstance(k, str) and k and (v is None or isinstance(v, str))
                    for k, v in criteria.items()
                )
            else:
                valid = isinstance(criteria, list) and len(criteria) >= 2 and all(
                    isinstance(v, str) and v for v in criteria
                ) and len(set(criteria)) == len(criteria)
            if not valid:
                _bad("choice question " + qid + " needs at least two named criteria")
        elif not (isinstance(criteria, list) and len(criteria) >= 2 and all(
            isinstance(v, str) and v for v in criteria
        )):
            _bad("score question " + qid + " needs at least two ordered criteria")
    model_id = payload.get("model", "laya-auto")
    if not isinstance(model_id, str) or model_id not in MODEL_IDS:
        _bad("Unknown model; use one of: " + ", ".join(MODEL_IDS))
    return state, questions, model_id


class _Backend:
    def __init__(self, router_factory: Callable[[], Any]):
        self._factory = router_factory
        self._router = None
        self._lock = threading.Lock()
        self._loaded_snapshot: list[str] = []

    def loaded(self) -> list[str]:
        return list(self._loaded_snapshot)

    def predict(self, state: Any, questions: dict, model: Optional[str]) -> dict:
        # Covers first construction, lazy checkpoint loading, and forward passes.
        with self._lock:
            if self._router is None:
                self._router = self._factory()
            try:
                return self._router.predict(state, questions, model=model)
            finally:
                self._loaded_snapshot = list(self._router.loaded)


def create_app(router_factory: Optional[Callable[[], Any]] = None, api_key: Optional[str] = None) -> FastAPI:
    """Create an app; router_factory is injectable for tests, without model weights."""
    if router_factory is None:
        def router_factory() -> Any:
            from laya import Router  # keep optional HTTP users and startup free of weights

            return Router(device=os.getenv("LAYA_DEVICE") or None,
                          max_loaded=int(os.getenv("LAYA_MAX_LOADED", "1")))

    backend = _Backend(router_factory)
    key = api_key if api_key is not None else os.getenv("LAYA_API_KEY")
    app = FastAPI(title="Laya HTTP API", version="0.3.6")
    app.state.backend = backend

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": {
            "message": str(exc.detail), "type": "invalid_request_error" if exc.status_code < 500 else "server_error"
        }})

    async def run_prediction(state: Any, questions: dict, model_id: str) -> dict:
        try:
            return await run_in_threadpool(backend.predict, state, questions, MODEL_IDS[model_id])
        except ValueError as exc:
            _bad(str(exc))
        except Exception:
            logging.exception("Laya inference failed")
            raise HTTPException(status_code=503, detail="Laya inference is unavailable") from None

    def authorize(authorization: Optional[str] = Header(default=None)) -> None:
        if not key:
            return
        if not authorization or not authorization.startswith("Bearer ") or not hmac.compare_digest(
            authorization[7:].encode("utf-8"), key.encode("utf-8")
        ):
            raise HTTPException(status_code=401, detail="Invalid bearer token")

    @app.get("/health")
    def health() -> dict:
        loaded = backend.loaded()
        return {"live": True, "model_loaded": bool(loaded), "loaded_models": loaded}

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    def models() -> dict:
        return {"object": "list", "data": [
            {"id": model_id, "object": "model", "owned_by": "laya"} for model_id in MODEL_IDS
        ]}

    @app.post("/v1/predict", dependencies=[Depends(authorize)])
    async def predict(payload: dict) -> dict:
        state, questions, model_id = _question_payload(payload)
        return await run_prediction(state, questions, model_id)

    @app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
    async def chat_completions(payload: dict) -> dict:
        if not isinstance(payload, dict):
            _bad("Request body must be a JSON object")
        unsupported = set(payload) - {"model", "messages", "stream"}
        if unsupported:
            _bad("Unsupported chat fields: " + ", ".join(sorted(unsupported)))
        if payload.get("stream", False) is not False:
            _bad("Streaming is unsupported")
        messages = payload.get("messages")
        if not isinstance(messages, list) or len(messages) != 1 or not isinstance(messages[0], dict):
            _bad("Exactly one user message is required")
        message = messages[0]
        if set(message) != {"role", "content"} or message["role"] != "user" or not isinstance(message["content"], str):
            _bad("Only one user message with JSON string content is supported")
        try:
            decision = json.loads(message["content"])
        except json.JSONDecodeError:
            _bad("User message content must be a JSON object string")
        if not isinstance(decision, dict):
            _bad("User message content must be a JSON object string")
        if "model" in decision:
            _bad("Set model in the chat request, not in message content")
        decision["model"] = payload.get("model", "laya-auto")
        state, questions, model_id = _question_payload(decision)
        result = await run_prediction(state, questions, model_id)
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)},
                         "finish_reason": "stop"}],
        }

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Laya typed decisions over HTTP")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.host not in ("127.0.0.1", "::1", "localhost") and not os.getenv("LAYA_API_KEY"):
        parser.error("Set LAYA_API_KEY before binding outside loopback")
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
