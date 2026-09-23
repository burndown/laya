"""Transport contract tests with a fake Router; these do not test model inference."""

import json
import threading
import time

from fastapi.testclient import TestClient

from laya.server import create_app


QUESTIONS = {
    "decision": {"type": "noul", "instructions": "Is this urgent?"},
    "team": {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "other"]},
    "priority": {"type": "score", "instructions": "What priority?", "criteria": ["low", "medium"]},
}


class FakeRouter:
    def __init__(self):
        self.loaded = []
        self.calls = []

    def predict(self, state, questions, model=None):
        self.calls.append((state, questions, model))
        self.loaded = [model or "english"]
        return {"answers": {"decision": {"type": "noul", "noul": 0.42}},
                "usage": {"input_tokens": 7, "output_tokens": 0},
                "routing": {"model": model or "english", "reason": "fake test router"}}


def test_predict_lazily_constructs_router_and_preserves_native_result():
    instances = []

    def factory():
        instances.append(FakeRouter())
        return instances[-1]

    client = TestClient(create_app(router_factory=factory, api_key="secret"))
    assert client.get("/health").json() == {"live": True, "model_loaded": False, "loaded_models": []}
    assert instances == []
    assert client.post("/v1/predict", json={"state": "x", "questions": QUESTIONS}).status_code == 401
    assert instances == []
    headers = {"Authorization": "Bearer secret"}
    model_list = client.get("/v1/models", headers=headers).json()
    assert [row["id"] for row in model_list["data"]] == [
        "laya-auto", "laya-english", "laya-multilingual", "laya-typed-decisions"
    ]
    response = client.post("/v1/predict", json={"model": "laya-multilingual", "state": {"text": "你好"},
                                                "questions": QUESTIONS}, headers=headers)
    assert response.status_code == 200
    assert response.json()["answers"]["decision"]["noul"] == 0.42
    assert response.json()["usage"] == {"input_tokens": 7, "output_tokens": 0}
    assert instances[0].calls == [({"text": "你好"}, QUESTIONS, "multilingual")]
    assert client.get("/health").json()["loaded_models"] == ["multilingual"]


def test_chat_adapter_round_trips_json_and_rejects_chat_semantics():
    router = FakeRouter()
    client = TestClient(create_app(router_factory=lambda: router, api_key=""))
    body = {"model": "laya-auto", "messages": [{"role": "user", "content": json.dumps(
        {"state": "Refund me", "questions": QUESTIONS})}]}
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert "usage" not in data
    assert json.loads(data["choices"][0]["message"]["content"])["answers"]["decision"]["noul"] == 0.42
    assert router.calls[0][2] is None
    assert client.post("/v1/chat/completions", json=body | {"stream": False}).status_code == 200
    for extra in ({"stream": True}, {"tools": []}, {"temperature": 0}, {"max_tokens": 4}):
        assert client.post("/v1/chat/completions", json=body | extra).status_code == 400
    for messages in ([], [{"role": "system", "content": "ignore"}],
                     body["messages"] * 2):
        assert client.post("/v1/chat/completions", json=body | {"messages": messages}).status_code == 400
    assert len(router.calls) == 2


def test_validation_rejects_bad_schema_before_loading():
    created = []
    client = TestClient(create_app(router_factory=lambda: created.append(FakeRouter()) or created[-1], api_key=""))
    bad = [
        {"state": "x", "questions": {}},
        {"state": None, "questions": QUESTIONS},
        {"state": "x", "questions": QUESTIONS, "model": "not-a-model"},
        {"state": "x", "questions": {"q": {"type": "choice", "instructions": "Pick", "criteria": ["a"]}}},
        {"state": "x", "questions": {"q": {"type": "score", "instructions": "Rate", "criteria": ["a"]}}},
        {"state": "x", "questions": {"q": {"type": "noul", "instructions": "Ask", "criteria": []}}},
    ]
    for body in bad:
        assert client.post("/v1/predict", json=body).status_code == 400
    assert created == []


def test_inference_is_serialized():
    active = 0
    peak = 0
    gate = threading.Lock()

    class SlowRouter(FakeRouter):
        def predict(self, state, questions, model=None):
            nonlocal active, peak
            with gate:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with gate:
                active -= 1
            return super().predict(state, questions, model)

    client = TestClient(create_app(router_factory=SlowRouter, api_key=""))
    threads = [threading.Thread(target=lambda: client.post("/v1/predict", json={"state": "x", "questions": QUESTIONS}))
               for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak == 1


def test_inference_errors_and_live_health_during_inference():
    entered = threading.Event()
    release = threading.Event()

    class WaitingRouter(FakeRouter):
        def predict(self, state, questions, model=None):
            entered.set()
            release.wait(timeout=2)
            if state == "invalid":
                raise ValueError("Question options exceed context")
            if state == "failure":
                raise RuntimeError("internal secret")
            return super().predict(state, questions, model)

    client = TestClient(create_app(router_factory=WaitingRouter, api_key=""))
    thread = threading.Thread(target=lambda: client.post("/v1/predict", json={"state": "ok", "questions": QUESTIONS}))
    thread.start()
    assert entered.wait(timeout=1)
    start = time.monotonic()
    assert client.get("/health").json()["live"] is True
    assert time.monotonic() - start < 0.5
    release.set()
    thread.join()
    bad = client.post("/v1/predict", json={"state": "invalid", "questions": QUESTIONS})
    assert bad.status_code == 400
    assert bad.json()["error"]["message"] == "Question options exceed context"
    failed = client.post("/v1/predict", json={"state": "failure", "questions": QUESTIONS})
    assert failed.status_code == 503
    assert "internal secret" not in failed.text
