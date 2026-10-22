# llm — Agent instructions

## Project status

Built and smoke-tested on macOS (server + menu bar app). Ready to deploy on Windows. Single-file server (`llm.py`).

## Target

- **OS**: macOS and Windows (cross-platform)
- **Single script**: `llm.py` (the server)
- **Wrappers**:
  - Windows: `%USERPROFILE%\llm.bat` calling `llm.py`
  - macOS/Linux: `llm` shell script (activates `.venv`, runs `~/.config/llm/llm.py`)
- **macOS extras**: `llm_tray.py` (rumps menu bar app), `llm.app` / `llm_service.app` bundles
- **Config dir**: `~/.config/llm/` (macOS/Linux) or `%USERPROFILE%\.config\llm\` (Windows)
  - `config.json` — server options (port, host, idle_timeout, load_timeout, llama_log_level, llama-server binary path). Set `host` to `"0.0.0.0"` to accept remote connections; `"127.0.0.1"` for local-only.
  - `models.json` — array of model definitions, each with `name`, `path`, and `params` (llama.cpp CLI flags per model)
- **Port**: 11434
- **Dependencies**: Python 3.10+, `pip install aiohttp`; macOS menu bar app also needs `rumps`

## Architecture

Proxy server that manages the configured llama.cpp executable as a subprocess:

1. Spawns the configured executable, `llama-params`, and model arguments (for example, `llama.exe serve -m <gguf> --host 127.0.0.1 --port <random>`) on first request for a model
2. Kills previous instance when switching models
3. Idle timeout unloads model after N seconds of inactivity
4. Streams requests/responses through to the child process

## Endpoints required

| Endpoint | Format | Notes |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI | For opencode, pi.dev, etc. |
| `GET /v1/models` | OpenAI | Returns configured models |
| `POST /api/chat` | Ollama | Streaming NDJSON: `{ message: { content }, done }` |
| `POST /api/generate` | Ollama | Ollama generate endpoint |
| `POST /api/show` | Ollama | Returns model info including context length. Respond with `{ model_info: { "llama.context_length": <ctx-size> }, parameters: "num_ctx <ctx-size>" }` using the model's `params.ctx-size` or a fallback of 4096. Required by ollamaChat. |
| `GET /api/tags` | Ollama | Returns `{ models: [{ name, size? }] }` — names from model config, size from file when present |
| `GET /api/version` | Ollama | Returns `{ version }` |

## Cross-platform

Developed on macOS, target is Windows. Differences to handle:

| Concern | macOS | Windows |
|---|---|---|
| Binary name | `llama-server` | `llama-server.exe` |
| Config dir | `~/.config/llm/` | `%USERPROFILE%\.config\llm\` |
| Process kill | `.terminate()` / `SIGTERM` | `.terminate()` → `TerminateProcess` (abrupt but acceptable for idle timeout / model swap) |
| Wrapper | `llm` shell script | `llm.bat` (generate both) |

Use `pathlib.Path.home() / ".config" / "llm"` for cross-platform config path resolution.

## opencode integration

Configured as a custom OpenAI-compatible provider in `opencode.json` using `@ai-sdk/openai-compatible` with `baseURL: http://localhost:11434/v1`. Each model in `models.json` gets its own entry under `provider.models`. Run `/models` in opencode to switch between them.
