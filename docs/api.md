# HTTP API

The optional FastAPI server exposes Laya's typed decisions over HTTP. It uses the same
`Router.predict(state, questions, model=...)` result, including `answers`, `usage`, and
`routing`. It does not generate conversational text.

```bash
python -m pip install -e '.[server]'
laya-serve --host 127.0.0.1 --port 8000
```

The server starts without downloading weights. The first prediction loads its checkpoint;
that request can take substantially longer. Inference is serialized within one process to
avoid concurrent model loading and memory spikes. `LAYA_DEVICE` and `LAYA_MAX_LOADED`
configure `Router(device=..., max_loaded=...)`; the default retains one checkpoint.
`GET /health` reports `live` separately from `model_loaded` and `loaded_models`.
`GET /v1/models` lists the four public model IDs:

| API model | Router `model` |
| --- | --- |
| `laya-auto` | `None` (language routing) |
| `laya-english` | `english` |
| `laya-multilingual` | `multilingual` |
| `laya-typed-decisions` | `typed-decisions` |

`POST /v1/predict` takes a JSON object with `state`, `questions`, and optional `model`:

```bash
curl -s http://127.0.0.1:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"model":"laya-auto","state":{"message":"Please refund the duplicate charge"},"questions":{"refund":{"type":"noul","instructions":"Does the customer request a refund?"},"team":{"type":"choice","instructions":"Which team handles this?","criteria":{"billing":"Payments and refunds","other":"Other requests"}},"urgency":{"type":"score","instructions":"How urgent?","criteria":["low","medium","high"]}}}'
```

Question IDs map to `choice` (two or more named criteria), `score` (two or more ordered
criteria), or `noul` (no criteria). `state` may be a string, JSON object, or JSON array.
The response is the native Router result; no probabilities are calculated by this adapter.

`POST /v1/chat/completions` is a narrow transport for clients that only support the
OpenAI chat API. Send exactly one `user` message. Its string content must be JSON
`{"state": ..., "questions": ...}`. Choose the Laya model in the outer `model` field.
The assistant message content is the JSON-encoded native Router result. Streaming,
tools, system prompts, earlier messages, temperature, and other chat controls are
rejected explicitly because Laya does not implement those semantics (`stream: false` is accepted). The chat envelope
does not claim token usage; the native Laya `usage` remains inside message content.

```python
import json
from openai import OpenAI

# Install this client separately: python -m pip install openai
# First use may download model weights, so allow a long initial timeout.
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local", timeout=180.0)
reply = client.chat.completions.create(
    model="laya-auto",
    messages=[{"role": "user", "content": json.dumps({
        "state": {"message": "Please refund the duplicate charge"},
        "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
    })}],
)
result = json.loads(reply.choices[0].message.content)
print(result["answers"]["refund"])
```

By default the CLI binds only to loopback and does not require a key. To bind to a
network interface, set `LAYA_API_KEY` first; requests to `/v1/*` then require
`Authorization: Bearer <key>`. The health endpoint remains unauthenticated.

Install `laya[server-test]` and run `pytest -q tests/test_server.py` to test the API
with an injected fake Router. Those tests do not download or evaluate model weights.
