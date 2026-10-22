#!/usr/bin/env python3
"""llm - Lightweight Ollama-compatible server for llama.cpp.

Usage:
  llm.py serve [--port PORT]
"""

import argparse
import asyncio
import errno
import json
import logging
import os
import signal
import socket
import sys
import time
from pathlib import Path

import aiohttp
from aiohttp import web
from aiohttp.client_exceptions import ClientConnectionResetError

logger = logging.getLogger("llm")

CONFIG_DIR = Path.home() / ".config" / "llm"
DEFAULT_CONFIG = {
    "llama-server": "llama-server",
    "llama-params": [],
    "port": 11434,
    "host": "0.0.0.0",
    "idle_timeout": 300,
}


def load_config():
    config_path = CONFIG_DIR / "config.json"
    models_path = CONFIG_DIR / "models.json"

    config = dict(DEFAULT_CONFIG)
    if config_path.exists():
        with open(config_path) as f:
            config.update(json.load(f))

    models = []
    if models_path.exists():
        with open(models_path) as f:
            models = json.load(f)

    return config, models


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def model_map_from_list(models):
    return {m["name"]: m for m in models}


def ollama_chat_body_to_openai(body):
    return {
        "model": body.get("model", ""),
        "messages": body.get("messages", []),
        "stream": body.get("stream", False),
    }


def _upstream_timing(data):
    """Extract token counts and durations from upstream response.
    Non-streaming has 'usage', streaming final chunk has 'timings'."""
    usage = data.get("usage", {})
    timings = data.get("timings", {})
    pc = usage.get("prompt_tokens") or timings.get("prompt_n") or 0
    ec = usage.get("completion_tokens") or timings.get("predicted_n") or 0
    p_ms = timings.get("prompt_ms", 0)
    e_ms = timings.get("predicted_ms", 0)
    out = {
        "prompt_eval_count": pc,
        "eval_count": ec,
        "prompt_eval_duration": int(p_ms * 1_000_000) if p_ms else 0,
        "eval_duration": int(e_ms * 1_000_000) if e_ms else 0,
        "total_duration": int((p_ms + e_ms) * 1_000_000) if (p_ms or e_ms) else 0,
    }
    return out


def openai_chunk_to_ollama(data):
    choice = data.get("choices", [{}])[0]
    delta = choice.get("delta", {})
    finish = choice.get("finish_reason")
    content = delta.get("content", "")
    role = delta.get("role", "assistant")
    is_done = finish is not None

    out = {
        "message": {"role": role, "content": content},
        "done": is_done,
    }
    if is_done:
        out["done_reason"] = finish
        out.update(_upstream_timing(data))
    return out


def openai_full_to_ollama(data, model_name):
    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    return {
        "model": model_name,
        "message": {
            "role": msg.get("role", "assistant"),
            "content": msg.get("content", ""),
        },
        "done": True,
        "done_reason": choice.get("finish_reason", "stop"),
        **_upstream_timing(data),
    }


def openai_chunk_to_ollama_generate(data):
    choice = data.get("choices", [{}])[0]
    delta = choice.get("delta", {})
    finish = choice.get("finish_reason")
    content = delta.get("content", "")
    is_done = finish is not None

    out = {"response": content, "done": is_done}
    if is_done:
        out["done_reason"] = finish
        out.update(_upstream_timing(data))
    return out


def openai_full_to_ollama_generate(data, model_name):
    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    return {
        "model": model_name,
        "response": msg.get("content", ""),
        "done": True,
        "done_reason": choice.get("finish_reason", "stop"),
        **_upstream_timing(data),
    }


class ModelProcess:
    def __init__(self, config, models):
        self.config = config
        self.models = models
        self._model_index = model_map_from_list(models)
        self._process = None
        self._port = None
        self._current_model = None
        self._idle_timer = None
        self._lock = asyncio.Lock()
        self._ready_event = asyncio.Event()
        self._active_requests = 0

    def find_model(self, name):
        return self._model_index.get(name)

    @property
    def current(self):
        return self._current_model

    async def ensure_model(self, model_name):
        async with self._lock:
            if self._current_model == model_name and self._is_alive():
                self._reset_idle()
                return self._port

            model = self.find_model(model_name)
            if not model:
                raise ValueError(f"Unknown model: {model_name}")

            await self._stop_locked()
            port = find_free_port()
            cmd = self._build_cmd(model, port)
            logger.info("Spawning %s on port %d", model_name, port)
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            await self._wait_ready(port)
            self._port = port
            self._current_model = model_name
            self._reset_idle()
            asyncio.create_task(self._pipe_stderr())
            return port

    def _is_alive(self):
        return self._process is not None and self._process.returncode is None

    def _build_cmd(self, model, port):
        binary = self.config["llama-server"]
        llama_params = self.config.get("llama-params", [])
        if not isinstance(llama_params, list) or not all(
            isinstance(param, str) for param in llama_params
        ):
            raise ValueError('"llama-params" must be an array of strings')
        cmd = [
            binary,
            *llama_params,
            "-m",
            model["path"],
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        for key, value in model.get("params", {}).items():
            flag = key if key.startswith("-") else f"--{key}"
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])
        return cmd

    async def _wait_ready(self, port, timeout=60):
        url = f"http://127.0.0.1:{port}/health"
        deadline = time.monotonic() + timeout
        last_error = None
        while time.monotonic() < deadline:
            try:
                async with aiohttp.ClientSession() as sess:
                    async with sess.get(url, timeout=aiohttp.ClientTimeout(total=2)):
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = exc
                await asyncio.sleep(0.5)
        raise TimeoutError(f"Model not ready after {timeout}s — {last_error}")

    async def stop(self):
        async with self._lock:
            await self._stop_locked()

    async def reset(self):
        """Force-stop the current model so the next request respawns a fresh one."""
        logger.warning("Resetting model process (self-heal)")
        async with self._lock:
            await self._stop_locked()

    async def health_ok(self):
        """Return True if the current upstream answers /health, False otherwise."""
        if not self._is_alive() or self._port is None:
            return False
        url = f"http://127.0.0.1:{self._port}/health"
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            return False

    async def reset_if_unhealthy(self):
        """Reset the model only if it is dead or unresponsive, so a single bad
        request does not tear down a working server."""
        if self._current_model is None:
            return
        if not self._is_alive() or not await self.health_ok():
            await self.reset()

    async def watchdog(self):
        """Periodically verify the loaded model is responsive and reset it if it
        gets stuck (e.g. after a transient network/system hiccup)."""
        interval = self.config.get("watchdog_interval", 15)
        threshold = self.config.get("watchdog_failures", 3)
        fails = 0
        while True:
            await asyncio.sleep(interval)
            if self._current_model is None or not self._is_alive():
                fails = 0
                continue
            if await self.health_ok():
                fails = 0
                continue
            fails += 1
            logger.warning("Upstream health check failed (%d/%d)", fails, threshold)
            if fails >= threshold:
                logger.error("Upstream unresponsive; resetting model")
                await self.reset()
                fails = 0

    async def _stop_locked(self):
        if self._idle_timer:
            self._idle_timer.cancel()
            self._idle_timer = None
        if self._is_alive():
            logger.info("Stopping model %s", self._current_model)
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                logger.warning("Killing unresponsive process")
                self._process.kill()
                await self._process.wait()
        if self._process and self._process.stderr and hasattr(self._process.stderr, "close"):
            try:
                self._process.stderr.close()
            except OSError:
                pass
        self._process = None
        self._port = None
        self._current_model = None

    def acquire(self):
        self._active_requests += 1
        self._cancel_idle()

    def release(self):
        self._active_requests -= 1
        if self._active_requests < 0:
            self._active_requests = 0
        self._reset_idle()

    def _cancel_idle(self):
        if self._idle_timer:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _reset_idle(self):
        self._cancel_idle()
        if self._active_requests > 0:
            return
        timeout = self.config.get("idle_timeout", 300)
        if timeout > 0:
            loop = asyncio.get_running_loop()
            self._idle_timer = loop.call_later(
                timeout, lambda: asyncio.create_task(self.stop())
            )

    async def _pipe_stderr(self):
        pipe = self._process.stderr
        if not pipe:
            return
        try:
            while True:
                line = await pipe.readline()
                if not line:
                    break
                logger.warning("[llama-server] %s", line.decode().strip())
        except Exception:
            pass
        finally:
            if hasattr(pipe, "close"):
                try:
                    pipe.close()
                except OSError:
                    pass


class Server:
    def __init__(self, config, models):
        self.config = config
        self.models = models
        self.mp = ModelProcess(config, models)

    async def handle_api_version(self, request):
        return web.json_response({"version": "0.1.0"})

    async def handle_v1_models(self, request):
        data = [
            {"id": m["name"], "object": "model", "created": int(time.time()), "owned_by": "llm"}
            for m in self.models
        ]
        return web.json_response({"object": "list", "data": data})

    async def handle_api_tags(self, request):
        models_list = []
        for m in self.models:
            info = {"name": m["name"]}
            p = m.get("path", "")
            if os.path.exists(p):
                info["size"] = os.path.getsize(p)
            models_list.append(info)
        return web.json_response({"models": models_list})

    async def handle_api_show(self, request):
        body = await request.json()
        model_name = body.get("model", "")
        model = self.mp.find_model(model_name)
        if not model:
            return web.json_response({"error": f"model '{model_name}' not found"}, status=404)
        ctx = model.get("params", {}).get("ctx-size", 4096)
        return web.json_response({
            "model_info": {"llama.context_length": ctx},
            "parameters": f"num_ctx {ctx}",
            "model": {"name": model["name"]},
        })

    async def _ensure(self, model_name):
        try:
            return await self.mp.ensure_model(model_name)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=404)
        except (TimeoutError, Exception) as exc:
            logger.exception("Model '%s' failed to load", model_name)
            return web.json_response(
                {"error": f"model '{model_name}' failed to load: {exc}"},
                status=503)

    async def handle_v1_chat(self, request):
        body = await request.json()
        return await self._proxy(request, body, "/v1/chat/completions")

    async def _call_upstream(self, sess, url, body):
        """POST to upstream via session, retrying on transient 503 loading errors."""
        for attempt in range(120):
            upstream = await sess.post(url, json=body)
            if upstream.status != 503:
                return upstream
            err_text = await upstream.text()
            upstream.release()
            if "loading" not in err_text.lower():
                return upstream
            if attempt == 119:
                return upstream
            await asyncio.sleep(0.5)

    async def handle_api_chat(self, request):
        self.mp.acquire()
        try:
            body = await request.json()
            model_name = body.get("model", "")
            stream = body.get("stream", False)

            port = await self._ensure(model_name)
            if isinstance(port, web.Response):
                return port

            url = f"http://127.0.0.1:{port}/v1/chat/completions"
            openai_body = ollama_chat_body_to_openai(body)

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=15)
            ) as sess:
                upstream = await self._call_upstream(sess, url, openai_body)
                async with upstream:
                    if upstream.status != 200:
                        err_text = await upstream.text()
                        return web.json_response(
                            {"error": f"upstream error ({upstream.status}): {err_text}"},
                            status=upstream.status,
                        )
                    if stream:
                        return await self._stream_translate(
                            request, upstream, openai_chunk_to_ollama
                        )
                    data = await upstream.json()
                    return web.json_response(openai_full_to_ollama(data, model_name))
        except (ConnectionResetError, ClientConnectionResetError) as exc:
            logger.debug("Client disconnected during api/chat: %s", exc)
            return web.json_response({"error": "connection reset"}, status=499)
        except (asyncio.TimeoutError, aiohttp.ClientError, OSError) as exc:
            logger.error("Upstream error in api/chat: %s", exc)
            await self.mp.reset_if_unhealthy()
            return web.json_response({"error": f"upstream error: {exc}"}, status=502)
        finally:
            self.mp.release()

    async def handle_api_generate(self, request):
        self.mp.acquire()
        try:
            body = await request.json()
            model_name = body.get("model", "")
            prompt = body.get("prompt", "")
            stream = body.get("stream", False)

            port = await self._ensure(model_name)
            if isinstance(port, web.Response):
                return port

            url = f"http://127.0.0.1:{port}/v1/chat/completions"
            openai_body = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": stream,
            }

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=15)
            ) as sess:
                upstream = await self._call_upstream(sess, url, openai_body)
                async with upstream:
                    if upstream.status != 200:
                        err_text = await upstream.text()
                        return web.json_response(
                            {"error": f"upstream error ({upstream.status}): {err_text}"},
                            status=upstream.status,
                        )
                    if stream:
                        return await self._stream_translate(
                            request, upstream, openai_chunk_to_ollama_generate
                        )
                    data = await upstream.json()
                    return web.json_response(openai_full_to_ollama_generate(data, model_name))
        except (ConnectionResetError, ClientConnectionResetError) as exc:
            logger.debug("Client disconnected during api/generate: %s", exc)
            return web.json_response({"error": "connection reset"}, status=499)
        except (asyncio.TimeoutError, aiohttp.ClientError, OSError) as exc:
            logger.error("Upstream error in api/generate: %s", exc)
            await self.mp.reset_if_unhealthy()
            return web.json_response({"error": f"upstream error: {exc}"}, status=502)
        finally:
            self.mp.release()

    async def _proxy(self, request, body, path):
        self.mp.acquire()
        try:
            model_name = body.get("model", "")
            stream = body.get("stream", False)

            port = await self._ensure(model_name)
            if isinstance(port, web.Response):
                return port
            url = f"http://127.0.0.1:{port}{path}"

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=15)
            ) as sess:
                async with sess.post(url, json=body) as upstream:
                    resp = web.StreamResponse(
                        status=upstream.status,
                        headers={
                            "Content-Type": upstream.headers.get(
                                "Content-Type", "text/event-stream" if stream else "application/json"
                            ),
                        },
                    )
                    await resp.prepare(request)
                    async for chunk in upstream.content.iter_any():
                        try:
                            await resp.write(chunk)
                        except (ConnectionResetError, ClientConnectionResetError):
                            return resp
                    return resp
        except (ConnectionResetError, ClientConnectionResetError) as exc:
            logger.debug("Client disconnected during proxy: %s", exc)
            return web.json_response({"error": "connection reset"}, status=499)
        except (asyncio.TimeoutError, aiohttp.ClientError, OSError) as exc:
            logger.error("Upstream error in proxy: %s", exc)
            await self.mp.reset_if_unhealthy()
            return web.json_response({"error": f"upstream error: {exc}"}, status=502)
        finally:
            self.mp.release()

    async def _stream_translate(self, request, upstream, translator):
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "application/x-ndjson", "Cache-Control": "no-cache"},
        )
        await resp.prepare(request)

        buf = b""
        try:
            async for chunk in upstream.content.iter_any():
                buf += chunk
                while b"\n\n" in buf:
                    raw, buf = buf.split(b"\n\n", 1)
                    for line in raw.split(b"\n"):
                        line = line.strip()
                        if not line or not line.startswith(b"data: "):
                            continue
                        payload = line[6:]
                        if payload.strip() == b"[DONE]":
                            continue
                        try:
                            openai_data = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        ollama_data = translator(openai_data)
                        try:
                            await resp.write((json.dumps(ollama_data) + "\n").encode())
                        except (ConnectionResetError, ClientConnectionResetError):
                            return resp
        except (ConnectionResetError, ClientConnectionResetError, aiohttp.ClientError, OSError):
            pass
        return resp


@web.middleware
async def cors_middleware(request, handler):
    if request.method == "OPTIONS":
        return web.Response(
            status=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
                "Access-Control-Max-Age": "86400",
            },
        )
    return await handler(request)


async def cors_headers(request, response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"


@web.middleware
async def error_middleware(request, handler):
    try:
        return await handler(request)
    except web.HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error processing %s %s", request.method, request.path)
        return web.json_response({"error": "internal server error"}, status=500)

def create_app(config, models):
    sv = Server(config, models)
    app = web.Application(middlewares=[cors_middleware, error_middleware])
    app.on_response_prepare.append(cors_headers)
    app.on_shutdown.append(lambda _: sv.mp.stop())

    async def _start_watchdog(app):
        app["watchdog"] = asyncio.create_task(sv.mp.watchdog())

    async def _stop_watchdog(app):
        task = app.get("watchdog")
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app.on_startup.append(_start_watchdog)
    app.on_cleanup.append(_stop_watchdog)

    app.router.add_get("/api/tags", sv.handle_api_tags)
    app.router.add_post("/api/tags", sv.handle_api_tags)
    app.router.add_get("/api/version", sv.handle_api_version)
    app.router.add_post("/api/show", sv.handle_api_show)
    app.router.add_post("/api/chat", sv.handle_api_chat)
    app.router.add_post("/api/generate", sv.handle_api_generate)
    app.router.add_get("/v1/models", sv.handle_v1_models)
    app.router.add_post("/v1/chat/completions", sv.handle_v1_chat)

    return app


# Transient OS network errors that must not crash the event loop: the interface
# went down/changed (sleep/wake, VPN, WiFi drop). We log and keep running.
_BENIGN_NET_ERRNOS = {
    errno.ENETDOWN, errno.ENETUNREACH, errno.ENETRESET, errno.ECONNABORTED,
    errno.EHOSTDOWN, errno.EHOSTUNREACH, errno.EADDRNOTAVAIL, errno.ENOTCONN,
    errno.EPIPE,
}


def _suppress_pipe_errors(loop, context):
    exc = context.get("exception")
    msg = context.get("message", "")
    if isinstance(exc, ConnectionResetError) and "connection_lost" in msg:
        logger.debug("Suppressed pipe transport error (benign Windows cleanup)")
        return
    if isinstance(exc, OSError) and exc.errno in _BENIGN_NET_ERRNOS:
        logger.warning("Transient network error in event loop (%s); continuing", exc)
        return
    loop.default_exception_handler(context)


async def _serve(config, models):
    """Run the server with a supervisor that rebinds the listening socket if the
    network device is unavailable (e.g. after a transient network failure)."""
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(_suppress_pipe_errors)

    app = create_app(config, models)
    runner = web.AppRunner(app)
    await runner.setup()
    host, port = config["host"], config["port"]

    stop = asyncio.Event()
    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass  # Windows / unsupported

    backoff = 1
    try:
        while not stop.is_set():
            site = web.TCPSite(runner, host, port, reuse_address=True)
            try:
                await site.start()
            except OSError as exc:
                logger.error(
                    "Failed to bind %s:%d (%s); retrying in %ds", host, port, exc, backoff
                )
                try:
                    await asyncio.wait_for(stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 30)
                continue
            logger.info("Listening on %s:%d", host, port)
            backoff = 1
            await stop.wait()
    finally:
        await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description="llm server for llama.cpp")
    parser.add_argument("command", nargs="?", default="serve", help="Command (serve)")
    parser.add_argument("--port", type=int, help="Override server port")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname).1s %(message)s")

    config, models = load_config()
    if not models:
        logger.warning("No models configured in %s", CONFIG_DIR / "models.json")
        sys.exit(1)

    if args.port:
        config["port"] = args.port

    logger.info("Starting llm on %s:%d", config["host"], config["port"])
    try:
        asyncio.run(_serve(config, models))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
