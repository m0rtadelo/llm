#!/usr/bin/env python3
"""llm_tray - macOS menu bar app for LLM server."""

import json
import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

import rumps

CONFIG_DIR = Path.home() / ".config" / "llm"
LLM_PY = CONFIG_DIR / "llm.py"
LLAMA_BIN = Path.home() / "Projects" / "m0rtadelo" / "llama.cpp" / "build" / "bin"


def _find_icon():
    for p in [
        Path(__file__).parent / "llm.app" / "Contents" / "Resources" / "iconTemplate.png",
        Path(__file__).parent / "Resources" / "iconTemplate.png",
    ]:
        if p.exists():
            return str(p)
    return None


class LLMTray(rumps.App):
    def __init__(self):
        icon = _find_icon()
        super().__init__("LLM", icon=icon)
        self.server_process = None
        self.venv_python = sys.executable
        self.server_port = self._load_port()

        self.status_item = rumps.MenuItem("Status: Stopped")
        self.toggle_item = rumps.MenuItem("Start Server", callback=self.toggle)

        self.menu = [
            self.status_item,
            None,
            self.toggle_item,
            None,
            rumps.MenuItem("Open Browser", callback=self._open_browser),
            None,
            rumps.MenuItem("Quit", callback=self._quit),
        ]

    def _load_port(self):
        try:
            cf = CONFIG_DIR / "config.json"
            if cf.exists():
                with open(cf) as f:
                    return json.load(f).get("port", 11434)
        except Exception:
            pass
        return 11434

    def toggle(self, _):
        if self.server_process and self.server_process.poll() is None:
            self._stop()
        else:
            self._start()

    def _start(self):
        if not LLM_PY.exists():
            rumps.alert("Error", f"llm.py not found at {LLM_PY}\nRun setup first.")
            return
        env = os.environ.copy()
        env["PATH"] = f"{LLAMA_BIN}:{env.get('PATH', '')}"
        try:
            self.server_process = subprocess.Popen(
                [str(self.venv_python), str(LLM_PY), "serve"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception as e:
            rumps.alert("Error", str(e))
            return
        if self.server_process.poll() is not None:
            err = self.server_process.stderr.read().decode().strip()
            rumps.alert("Server Failed", err or "Unknown error")
            self.server_process = None
            return
        self.status_item.title = "Status: Running"
        self.toggle_item.title = "Stop Server"
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _stop(self):
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
            self.server_process = None
        self.status_item.title = "Status: Stopped"
        self.toggle_item.title = "Start Server"

    def _drain_stderr(self):
        try:
            for line in iter(self.server_process.stderr.readline, b""):
                pass
        except Exception:
            pass

    def _open_browser(self, _):
        webbrowser.open(f"http://localhost:{self.server_port}")

    def _quit(self, _):
        self._stop()
        rumps.quit_application()


if __name__ == "__main__":
    LLMTray().run()
