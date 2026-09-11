"""
run.py - Dedicated Launcher for Urban Flood Early Warning & Safe Route Backend Service

Launches:
- FastAPI Backend Service (http://0.0.0.0:8000)
- Interactive Swagger API Docs (http://localhost:8000/docs)
- System Health Check (http://localhost:8000/health)

Usage:
    python run.py
"""

import os
import sys
import time
import socket
import signal
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def is_port_in_use(port: int) -> bool:
    """Checks if a local TCP port is already open and listening on IPv4 or IPv6."""
    for host in ["127.0.0.1", "localhost", "::1"]:
        try:
            with socket.socket(socket.AF_INET6 if host == "::1" else socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.3)
                if s.connect_ex((host, port)) == 0:
                    return True
        except Exception:
            pass
    return False


def main():
    print("=" * 65)
    print("Urban Flood Early Warning & Safe Route - Backend Service")
    print("=" * 65)

    backend_port = int(os.environ.get("PORT", 8000))
    backend_host = os.environ.get("HOST", "0.0.0.0")

    if is_port_in_use(backend_port):
        print(f"[!] Warning: Port {backend_port} is already in use.")
        print(f"    If the backend is already running, access it at: http://localhost:{backend_port}")
        return

    print(f"[+] Starting FastAPI Backend Service on http://{backend_host}:{backend_port} ...")
    cmd_backend = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        backend_host,
        "--port",
        str(backend_port),
    ]

    p_backend = subprocess.Popen(cmd_backend, cwd=str(ROOT_DIR))

    # Wait for backend to initialize and bind port
    for _ in range(30):
        time.sleep(1)
        if is_port_in_use(backend_port):
            print("[OK] FastAPI Backend is online and accepting requests!")
            break
    else:
        print("[!] Warning: Backend took longer than expected to bind port.")

    print("=" * 65)
    print("[*] Backend Service Endpoints:")
    print(f"   * API Root Endpoint           : http://localhost:{backend_port}/")
    print(f"   * Interactive Swagger Docs    : http://localhost:{backend_port}/docs")
    print(f"   * ReDoc API Reference         : http://localhost:{backend_port}/redoc")
    print(f"   * System Health Telemetry     : http://localhost:{backend_port}/health")
    print(f"   * Real-Time Flood Risk API    : http://localhost:{backend_port}/flood-risk")
    print(f"   * Live Weather Nowcast API    : http://localhost:{backend_port}/weather/live")
    print(f"   * Safe Emergency Route API    : http://localhost:{backend_port}/route/safe")
    print("=" * 65)
    print("CORS is configured to accept requests from any origin (allow_origins=['*']).")
    print("External frontends can connect directly to these endpoints.")
    print("Press Ctrl+C to terminate backend service.\n")

    def handle_sigint(sig, frame):
        print("\nStopping backend service...")
        try:
            p_backend.terminate()
            p_backend.wait(timeout=5)
        except Exception:
            p_backend.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        p_backend.wait()
    except KeyboardInterrupt:
        handle_sigint(None, None)


if __name__ == "__main__":
    main()
