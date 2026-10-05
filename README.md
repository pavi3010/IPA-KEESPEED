# DeepSeek API: a free LLM API powered by DeepSeek

**Using your own DeepSeek account.** No API key, no credits, no paid plan: it turns the free chat at [chat.deepseek.com](https://chat.deepseek.com) into an API you can call from code.

You can use it in two ways:

- 🐍 **As a Python library:** just call `client.chat("Hi")`. Supports streaming and multi-turn conversations.
- 🔌 **As a local OpenAI-compatible API:** runs a server at `http://localhost:8000/v1` that speaks the OpenAI format, so the official `openai` SDK (and any OpenAI-compatible app) works as a drop-in, with `localhost` in place of OpenAI.

You sign in once in a browser with your DeepSeek account; your session is saved and refreshed automatically after that.

> **Unofficial project.** Not affiliated with or endorsed by DeepSeek. It automates the consumer DeepSeek web experience for personal use, so use it responsibly and within DeepSeek's terms.

---

## Table of contents

- [Why use this?](#why-use-this)
- [Requirements](#requirements)
- [Setup (2 minutes)](#setup-2-minutes)
- [Usage 1: In Python (no server)](#usage-1-in-python-no-server)
- [Usage 2: As an OpenAI-compatible server](#usage-2-as-an-openai-compatible-server)
- [Command line](#command-line)
- [Human-check & proof-of-work (automatic)](#human-check--proof-of-work-automatic)
- [Models, DeepThink & web search](#models-deepthink--web-search)
- [Tool calling (function calling)](#tool-calling-function-calling)
- [Upstream errors](#upstream-errors)
- [Concurrency](#concurrency)
- [Rate limiting](#rate-limiting)
- [Project layout](#project-layout)
- [Notes & limitations](#notes--limitations)
- [License](#license)

---

## Why use this?

- **Free:** uses your normal signed-in DeepSeek account, no API billing.
- **Drop-in OpenAI replacement:** point any OpenAI client at `localhost` and it just works.
- **Full DeepSeek toolset:** pick the fast or expert model, and toggle DeepThink reasoning and web search per request.
- **Streaming + conversations:** token-by-token output and multi-turn threads addressed by `conversation_id`.

---

## Requirements

- **Python 3.9+**
- A **DeepSeek account** (the free one you use for [chat.deepseek.com](https://chat.deepseek.com) is fine)
- Works on Windows, macOS, and Linux

---

## Setup (2 minutes)

```bash
# 1. Clone the project
git clone https://github.com/sums001/Deepseek-API.git
cd "Deepseek-API"
```

**2. Create and activate a virtual environment**

On **macOS / Linux**:

```bash
python3 -m venv venv
source venv/bin/activate
```

On **Windows** (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

> On Windows you may need to allow script execution once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. In `cmd.exe` activate with `venv\Scripts\activate.bat` instead.

**3. Install dependencies and sign in**

```bash
# Install dependencies
pip install -r requirements.txt

# Install the browser Playwright needs (one-time)
playwright install chromium

# Sign in once: a browser opens, log into your DeepSeek account
python -m deepseek.auth
```

The login window opens so you can sign in by hand and solve the human-check once. After that your session (bearer token + cookies) is saved under `session/` (git-ignored, never shared) and reused on every run — the cached session is refreshed automatically, so your first request works right away.

> The server can also open this window for you on demand the first time it needs a session, so this step is optional for local single-user use.

---

## Usage 1: In Python (no server)

The simplest way if your code is already Python.

```python
from deepseek import DeepSeekClient

client = DeepSeekClient()                # loads your signed-in session

# Get a full reply
reply = client.chat("Say hello in one short sentence.")
print(reply.text)

# Continue the SAME conversation — pass the id back
reply2 = client.chat("And now in French?", conversation_id=reply.conversation_id)
print(reply2.text)

# Stream the answer as it's typed
for chunk in client.stream("Tell me a short joke"):
    print(chunk, end="", flush=True)
```

`chat()` returns the full text plus a `conversation_id`; pass that id back to keep the thread going, or omit it to start fresh. `stream()` yields the reply piece by piece.

👉 More: [examples/01_direct_chat.py](examples/01_direct_chat.py), [02_direct_conversation.py](examples/02_direct_conversation.py), [03_direct_stream.py](examples/03_direct_stream.py)

---

## Usage 2: As an OpenAI-compatible server

Start a local server that speaks the OpenAI API, so existing OpenAI tools and SDKs work unchanged.

```bash
python app.py
# -> DeepSeek OpenAI-compatible API on http://127.0.0.1:8000
```

Then point any OpenAI client at it (the API key is required by the SDK but ignored):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

Or call it with plain HTTP / `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-chat", "messages": [{"role": "user", "content": "Hello!"}]}'
```

**Endpoints** (both `/v1/...` and root `/...` supported)

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` or `/chat/completions` | Chat completions (stream, full OpenAI schema, tools, thinking, search) |
| `GET`  | `/v1/models` or `/models` | Lists available models and aliases (`deepseek-chat`, `gpt-4o`, etc.) |
| `GET`  | `/v1/models/{model}` or `/models/{model}` | Retrieve single model details |
| `POST` | `/v1/embeddings` or `/embeddings` | OpenAI-compatible embedding fallback |
| `GET`  | `/healthz` or `/` | Health check |

> Point your OpenAI client base URL to either `http://localhost:8000/v1` or `http://localhost:8000`. Both will route seamlessly.

> Change the address with env vars: `HOST=0.0.0.0 PORT=8080 python app.py`, or run `uvicorn server.api:app --host 0.0.0.0 --port 8080`.

👉 More: [examples/04_server_http.py](examples/04_server_http.py), [examples/05_server_stream.py](examples/05_server_stream.py), [examples/06_server_openai_sdk.py](examples/06_server_openai_sdk.py)

---

## Command line

```bash
python -m deepseek.auth          # sign in and save the session
```

---

## Human-check & proof-of-work (automatic)

DeepSeek's chat sits behind two gates, both handled for you:

- **AWS WAF human-check:** access needs a signed-in browser session that has
  cleared the "verify you're human" check. `python -m deepseek.auth` opens a real
  browser so you can sign in and solve it once; the resulting token + cookies are
  cached under `session/` and reused on every request.
- **Proof-of-work:** every completion is gated by a PoW challenge. The bridge
  solves it by running DeepSeek's own `sha3_wasm_bg.wasm` module — the same one
  the browser loads — inside a `wasmtime` sandbox, so there's nothing to do on
  your end.

A cached session is reused for ~6 hours and refreshed headlessly from your saved
Chrome profile when possible; only a full expiry sends you back to the browser.

---

## Models, DeepThink & web search

The `model` name selects **which model** answers. DeepThink and web search are
**not** models — they're orthogonal toggles you pass per request.

| Model | DeepSeek mode | Notes |
| --- | --- | --- |
| `deepseek-chat` | Instant | Fast default model |
| `deepseek-expert` | Expert | Stronger, slower |

Pass `thinking: true` (DeepThink reasoning) and/or `search: true` (web search) in
the request body — or via the OpenAI SDK's `extra_body`:

```python
resp = client.chat.completions.create(
    model="deepseek-expert",
    messages=[{"role": "user", "content": "What changed in the news today?"}],
    extra_body={"thinking": True, "search": True},
)
```

`conversation_id`, `thinking`, and `search` are non-OpenAI extras. A thread's
model is fixed at creation, so `model` can't be combined with `conversation_id`
on resume. Unknown model names return a `404` (no silent fallback). See
[server/config.py](server/config.py).

---

## Tool calling (function calling)

The server accepts OpenAI-style `tools` and returns `tool_calls`, so agent
frameworks that drive a tool loop work against it:

```python
resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
)
call = resp.choices[0].message.tool_calls[0]   # finish_reason == "tool_calls"
print(call.function.name, call.function.arguments)
```

Send the result back as a `tool` message (with the matching `tool_call_id`) and
the conversation continues as it would against OpenAI.

**This is emulated, not native.** DeepSeek's web chat has no function-calling
channel, so the bridge puts the tool schemas into the prompt with a strict
output contract and parses the model's JSON, DSML, or code reply back into `tool_calls`. That
has consequences worth knowing:

- **It's best-effort.** A prompt is not a decoder constraint. Complex or deeply
  nested parameter schemas are where it frays first.
- **DeepSeek DSML markup is natively parsed.** Tags like `<｜｜DSML｜｜ calls>...<｜｜DSML｜｜ invoke name="...">...`
  are intercepted, extracted into structured OpenAI `tool_calls`, and stripped from assistant text.
- **Unknown tool names in an explicit `{"tool_calls": ...}` envelope are passed
  through.** Clients with lazy tool loading (a tool-search step, MCP servers
  resolved on demand) legitimately call tools that aren't in the request's
  `tools` array, so the call goes to the client to resolve.
- **Other call dialects are salvaged.** Models don't reliably follow the
  contract, so replies that narrate the call ReAct-style (`Action: some_tool` /
  `Action Input: {...}`) or write it as code (`some_tool({"arg": 1})`) are
  parsed into real calls anyway. XML forms (`<some_tool arg="value">`, `<tool_call>...</tool_call>`)
  are handled too.
- **Python dict repr is accepted.** `{'tool_calls': [...]}` with single quotes
  and `True`/`None` is parsed via `ast.literal_eval`.
- **Nested wrappers are unwrapped.** A model that names the tool `tool_call` and
  nests the real call inside `arguments` still gets a working call.
- **Truncated replies are repaired structurally, never by guessing.** A cut-off
  object is closed so it can be parsed, but a half-written *value* is dropped
  with its key.
- **`tool_choice`** supports `"none"`, `"required"`, and pinning a named
  function.
- **Streaming buffers.** A tool call is only recognisable once its JSON / DSML is
  complete, so tool-enabled streaming requests emit one delta instead of
  token-by-token output. Requests without `tools` stream as before.
- **Replies are cut at the first role label.** The prompt serialises the
  conversation with `User:` / `Assistant:` / `Tool result (...)` labels, and everything
  after the first label is dropped to prevent hallucinated turns.

Set `DEBUG_REQUESTS=1` to log each request's message roles and offered tool
names — the quickest way to tell whether a client actually sent `tools`.

Tests: `python -m pytest tests/test_tool_calling.py` (no DeepSeek session
needed; the upstream client is faked).

---

## Upstream errors

DeepSeek reports failures inside the completion stream as an `event: hint`
frame, not an HTTP status:

```
event: hint
data: {"type":"error","content":"Messages too frequent. Try again later.",
       "finish_reason":"rate_limit_reached"}
```

Those frames are surfaced rather than dropped, because a dropped one becomes a
`200` with empty content — indistinguishable from the model having nothing to
say, with no reason to show and nothing for a client to back off from.

| Upstream | Response |
| --- | --- |
| `rate_limit_reached` | `429` + `Retry-After` (`UPSTREAM_RETRY_AFTER`, default 30) |
| any other error frame | `502` with the upstream message |

Streaming requests get the same status: the first delta is pulled before the
response starts, so an error still lands as a status code rather than arriving
mid-body after a `200`.

---

## Concurrency

The server bridges a **single** signed-in DeepSeek account behind one shared
client. The PoW solver's `wasmtime` store isn't reentrant, so upstream calls are
**serialized**: parallel HTTP requests queue behind a lock and run one at a time
(see [server/api.py](server/api.py)). This is intentional — throughput is
sequential, not parallel. Keep concurrent in-flight requests low, and please
don't hammer your account.

---

## Rate limiting

On top of serialization, the bridge enforces a self-imposed rate limit with a
dependency-free sliding-window limiter ([server/ratelimit.py](server/ratelimit.py)):
it caps accepted requests **per client IP** and returns a standard `429` +
`Retry-After` when you exceed it. `/healthz` is exempt.

| Env var | Default | Meaning |
| --- | --- | --- |
| `RATE_LIMIT_PER_MINUTE` | `30` | Requests/minute accepted per client IP |

```bash
RATE_LIMIT_PER_MINUTE=60 python app.py   # raise it
```

**On the client side, use exponential backoff.** Transient `429`s clear if you
retry with growing delays (e.g. 1s, 2s, 4s). The official `openai` SDK does this
automatically and honours `Retry-After`; with plain HTTP, add a few retries
yourself.

---

## Project layout

| Path | What it does |
| --- | --- |
| [deepseek/](deepseek/) | The core library: `DeepSeekClient`, auth/browser sign-in ([auth.py](deepseek/auth.py)), the HTTP driver ([client.py](deepseek/client.py)), and the PoW solver ([pow.py](deepseek/pow.py)) |
| [server/](server/) | The FastAPI OpenAI-compatible server |
| [examples/](examples/) | Runnable examples for every feature ([examples/README.md](examples/README.md)) |
| [app.py](app.py) | Starts the server |

---

## Notes & limitations

- **Sign in once, then reuse.** The cached session refreshes automatically; you only re-sign-in if it fully expires.
- **Be reasonable.** Please use it in moderation, and don't spam or hammer it with automated bulk requests.
- **No real token counts.** `usage` in responses is a rough ~4-chars/token estimate.
- **Most OpenAI params are accepted but ignored** (`temperature`, `top_p`, `max_tokens`); only `model`, `messages`, `stream`, `conversation_id`, `thinking`, and `search` do anything.
- **Vision is deferred.** It needs image-upload plumbing that isn't built yet.
- **Your session is private.** Everything in `session/` (cookies + token) stays on your machine and is git-ignored.

## License

Released under the [MIT License](LICENSE). As this is an unofficial project, you remain responsible for complying with DeepSeek's terms of service.

---

