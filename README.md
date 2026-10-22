# llm

A lightweight, Ollama- and OpenAI-compatible proxy server for [llama.cpp](https://github.com/ggml-org/llama.cpp). It exposes a single endpoint (default `http://localhost:11434`) that dynamically launches `llama-server` per model, swaps models on demand, and unloads them after an idle timeout.

Point any Ollama or OpenAI-compatible client (opencode, pi.dev, Continue, editors using `ollamaChat`, etc.) at it and switch between your local GGUF models without running a full Ollama installation.

## Features

- **Ollama-compatible API** — `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/version`.
- **OpenAI-compatible API** — `/v1/chat/completions`, `/v1/models`.
- **On-demand model loading** — spawns `llama-server` on a random free port on first request for a model.
- **Automatic model swapping** — kills the previous instance when a different model is requested.
- **Idle timeout** — unloads the active model after `idle_timeout` seconds of inactivity to free memory/VRAM.
- **Streaming** — NDJSON streaming for Ollama endpoints, SSE passthrough for OpenAI.
- **Per-model llama.cpp flags** — each model carries its own CLI parameters (context size, GPU layers, KV cache quant, flash attention, etc.).
- **CORS enabled** — works from browser-based clients.
- **Single file** — the whole server is [llm.py](llm.py); only dependency is `aiohttp`.

## Requirements

- Python 3.10+
- `pip install aiohttp`
- A built `llama-server` binary (from llama.cpp) on `PATH` or referenced by full path in the config; alternatively, configure `llama.exe` with the `serve` subcommand
- One or more GGUF model files
- macOS menu bar app only: `pip install rumps`

## Installation

The server reads its configuration from `~/.config/llm/` on both platforms:

- macOS/Linux: `~/.config/llm/`
- Windows: `%USERPROFILE%\.config\llm\`

1. Copy `llm.py` into the config directory:

   - macOS/Linux: `~/.config/llm/llm.py`
   - Windows: `%USERPROFILE%\.config\llm\llm.py`

2. Create `config.json` and `models.json` in that directory (see below).

3. Install the dependency:

   ```sh
   pip install aiohttp
   ```

### Configuration — `config.json`

```json
{
  "llama-server": "llama-server",
  "llama-params": [],
  "port": 11434,
  "host": "0.0.0.0",
  "idle_timeout": 300,
  "load_timeout": 600,
  "llama_log_level": "info"
}
```

| Key | Description |
|---|---|
| `llama-server` | Executable name or full path, such as `llama-server.exe` or `llama.exe` on Windows. |
| `llama-params` | Array of string arguments inserted immediately after the executable and before the model arguments. Use `["serve"]` with `llama.exe` to launch the `serve` subcommand. |
| `port` | Port the proxy listens on. Default `11434`. |
| `host` | `0.0.0.0` to accept remote connections; `127.0.0.1` for local-only. |
| `idle_timeout` | Seconds of inactivity before the active model is unloaded. `0` disables auto-unload. |
| `load_timeout` | Seconds to wait for `llama-server` to report healthy after spawning. Default `600`. |
| `llama_log_level` | Log level for `llama-server` stderr output: `info` (default), `debug`, `warning`, or `off` to silence it. |

### Configuration — `models.json`

An array of model definitions. Each `params` entry is passed to `llama-server` as a CLI flag (keys are prefixed with `--` unless they already start with `-`; boolean `true` becomes a bare flag).

```json
[
  {
    "name": "codellama",
    "path": "C:\\Users\\you\\models\\codellama-34b.Q4_K_M.gguf",
    "params": {
      "-ctk": "q4_0",
      "-ctv": "q4_0",
      "--flash-attn": "on",
      "--reasoning-budget": 256,
      "temp": 0.2,
      "ctx-size": 4096,
      "n-gpu-layers": 35
    }
  },
  {
    "name": "deepseek-coder",
    "path": "D:\\models\\deepseek-coder.gguf",
    "params": {
      "ctx-size": 8192,
      "n-gpu-layers": 50,
      "threads": 8,
      "temp": 0.1
    }
  }
]
```

The `ctx-size` param is also reported to clients via `/api/show` as the model context length (fallback `4096`).

Params are passed through verbatim, so they must exist in the `llama-server` build you run. If a flag was removed upstream, the process exits immediately and the proxy returns the child's own error. For example, recent llama.cpp builds replaced `--no-mmap` / `--mlock` with `--load-mode none|mlock|mmap|auto`.

## Running

Run directly with Python:

```sh
python ~/.config/llm/llm.py serve
```

Options:

- `--port PORT` — override the configured port.
- `--verbose` / `-v` — enable debug logging.

### Wrappers

- **Windows** — [llm.bat](llm.bat) calls the script from `%USERPROFILE%\.config\llm\llm.py`. Place it at `%USERPROFILE%\llm.bat` and run `llm serve`.
- **macOS/Linux** — the [llm](llm) shell script activates the project virtualenv and runs the script from `~/.config/llm/llm.py`.

### macOS menu bar app

[llm_tray.py](llm_tray.py) provides a `rumps`-based menu bar app to start/stop the server and open the browser. Two `.app` bundles are included:

- `llm_service.app` — launches the tray app ([llm_tray.py](llm_tray.py)).
- `llm.app` — launches the server directly (`llm.py serve`).

Both bundles prepend the llama.cpp build directory to `PATH` and activate the project virtualenv before starting. Adjust the paths inside `Contents/MacOS/llm` for your machine.

## API reference

### Ollama-compatible

| Endpoint | Method | Notes |
|---|---|---|
| `/api/chat` | POST | Streaming NDJSON (`{ message: { content }, done }`) or single JSON response. |
| `/api/generate` | POST | Prompt-based generation, streaming or single response. |
| `/api/tags` | GET/POST | Returns `{ models: [{ name, size? }] }` from your config. |
| `/api/show` | POST | Returns `{ model_info: { "llama.context_length" }, parameters }`. |
| `/api/version` | GET | Returns `{ version }`. |

### OpenAI-compatible

| Endpoint | Method | Notes |
|---|---|---|
| `/v1/chat/completions` | POST | Proxied to the upstream `llama-server`; streaming SSE passthrough. |
| `/v1/models` | GET | Returns configured models in OpenAI list format. |

## How it works

1. On the first request for a model, the proxy picks a random free port and spawns `llama-server -m <gguf> --host 127.0.0.1 --port <random>` with that model's params.
2. It waits for the upstream `/health` endpoint before forwarding the request, retrying while the model reports `loading`.
3. Requests are translated between the Ollama and OpenAI formats as needed and streamed back to the client.
4. When a different model is requested, the current `llama-server` process is terminated before the new one starts.
5. After `idle_timeout` seconds with no active requests, the running model is unloaded.

## opencode integration

Configure `llm` as a custom OpenAI-compatible provider in `opencode.json` using `@ai-sdk/openai-compatible` with `baseURL: http://localhost:11434/v1`. Add one entry per model under `provider.models`, then run `/models` in opencode to switch between them.

## Project layout

| File | Purpose |
|---|---|
| [llm.py](llm.py) | The proxy server (single file). |
| [config.json](config.json) | Example server configuration. |
| [models.json](models.json) | Example model definitions. |
| [llm.bat](llm.bat) | Windows wrapper. |
| [llm](llm) | macOS/Linux wrapper. |
| [llm_tray.py](llm_tray.py) | macOS menu bar app. |
| `llm.app/`, `llm_service.app/` | macOS `.app` launchers. |
| [AGENTS.md](AGENTS.md) | Notes for AI coding agents. |
