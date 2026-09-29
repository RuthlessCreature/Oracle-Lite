from __future__ import annotations

import socket
import threading
import webbrowser

import uvicorn

from .app import create_app


def _choose_port(start: int = 7870) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free localhost port found for Talker")


def main() -> None:
    port = _choose_port()
    url = f"http://127.0.0.1:{port}/"
    app = create_app()
    print(f"Oracle-Lite Talker {url}", flush=True)
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
