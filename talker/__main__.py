from __future__ import annotations

import ipaddress
import socket
import threading
import webbrowser

import psutil
import uvicorn

from .app import create_app


def _choose_port(start: int = 7870) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port found for Talker")


def _lan_ipv4_addresses() -> list[str]:
    """Return useful private LAN IPv4 addresses, preferring non-VPN interfaces."""
    preferred: list[str] = []
    fallback: list[str] = []
    vpnish = ("tun", "tap", "wg", "utun", "tailscale", "docker", "veth", "virbr")

    for interface, addresses in psutil.net_if_addrs().items():
        for address in addresses:
            if address.family != socket.AF_INET:
                continue
            try:
                ip = ipaddress.ip_address(address.address)
            except ValueError:
                continue
            if ip.is_loopback or ip.is_link_local or not ip.is_private:
                continue
            target = fallback if any(token in interface.lower() for token in vpnish) else preferred
            value = str(ip)
            if value not in target:
                target.append(value)

    return preferred + [ip for ip in fallback if ip not in preferred]


def main() -> None:
    port = _choose_port()
    loopback_url = f"http://127.0.0.1:{port}/"
    localhost_url = f"http://localhost:{port}/"
    lan_urls = [f"http://{ip}:{port}/" for ip in _lan_ipv4_addresses()]
    browser_url = lan_urls[0] if lan_urls else loopback_url

    app = create_app()
    print("Oracle-Lite Talker", flush=True)
    print(f"  Loopback: {loopback_url}", flush=True)
    print(f"  Localhost: {localhost_url}", flush=True)
    for url in lan_urls:
        print(f"  LAN:       {url}", flush=True)
    if lan_urls:
        print(
            "  Note: Talker is listening on all local interfaces so the LAN URL "
            "can bypass VPN/proxy handling of localhost.",
            flush=True,
        )
    print(
        "  Security: there is no authentication; do not expose this port to the Internet.",
        flush=True,
    )

    threading.Timer(1.0, lambda: webbrowser.open(browser_url)).start()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
