"""Legacy compatibility launcher.

The real application is the FastAPI app in `app/main.py`.
This file remains only so older local workflows still have an entrypoint.
"""

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent


def load_legacy_env() -> None:
    root_env = ROOT_DIR / ".env"
    if root_env.exists():
        load_dotenv(root_env)


def get_bind_host() -> str:
    return (os.getenv("HOST") or os.getenv("AUTH0_APP_HOST") or "0.0.0.0").strip()


def get_bind_port() -> int:
    raw = os.getenv("PORT") or os.getenv("AUTH0_APP_PORT") or "8000"
    return int(raw)


load_legacy_env()


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=get_bind_host(),
                port=get_bind_port(), reload=False)
