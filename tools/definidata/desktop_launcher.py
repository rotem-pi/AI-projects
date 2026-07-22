#!/usr/bin/env python3
"""Launches definiData as a native window (no browser tab): starts the
Streamlit server in the background and opens it in a plain OS webview window.

This is what the definiData.app bundle (built by install.sh) executes.
"""
import atexit
import os
import socket
import subprocess
import sys
import threading
import time

import webview

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = 8765


def wait_for_server(port: int, timeout: float = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def watch_and_close(proc: subprocess.Popen, window) -> None:
    """If the Streamlit server exits on its own (e.g. app.py called
    os._exit(0) after a self-update), close this window too so the whole
    app quits cleanly instead of showing a dead connection."""
    proc.wait()
    try:
        window.destroy()
    except Exception:
        pass


def main() -> None:
    env = os.environ.copy()
    env["DEFINIDATA_NATIVE"] = "1"

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", os.path.join(APP_DIR, "app.py"),
            "--server.port", str(PORT),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=APP_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(proc.terminate)

    if not wait_for_server(PORT):
        proc.terminate()
        raise RuntimeError("definiData server failed to start within 30s")

    window = webview.create_window(
        "definiData",
        f"http://localhost:{PORT}",
        width=1100,
        height=780,
        min_size=(700, 500),
        text_select=True,
    )
    threading.Thread(target=watch_and_close, args=(proc, window), daemon=True).start()
    webview.start()
    proc.terminate()


if __name__ == "__main__":
    main()
