import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from auth0_server_python.auth_server.server_client import ServerClient
from auth0_server_python.auth_types import StateData, TransactionData
from dotenv import load_dotenv


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent


def load_oauth_env() -> None:
    root_env = ROOT_DIR / ".env"
    if root_env.exists():
        load_dotenv(root_env)


load_oauth_env()


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"必須環境変数が未設定です: {name}")
    return value


STORE_DIR = Path(tempfile.gettempdir()) / "ps-socr-web-auth-store"
STATE_STORE_FILE = STORE_DIR / "state.json"
TRANSACTION_STORE_FILE = STORE_DIR / "transactions.json"


class JsonFileStore:
    """Simple JSON-backed store for local development."""

    def __init__(self, file_path: Path):
        self._file_path = file_path
        self._lock = threading.Lock()
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

    def _read_all(self) -> dict[str, dict]:
        if not self._file_path.exists():
            return {}
        try:
            return json.loads(self._file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_all(self, data: dict[str, dict]) -> None:
        self._file_path.write_text(
            json.dumps(data, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    def get_raw(self, key: str) -> dict | None:
        with self._lock:
            return self._read_all().get(key)

    def set_raw(self, key: str, value: dict) -> None:
        with self._lock:
            data = self._read_all()
            data[key] = value
            self._write_all(data)

    def delete_raw(self, key: str) -> None:
        with self._lock:
            data = self._read_all()
            if key in data:
                data.pop(key, None)
                self._write_all(data)

    def iter_items(self) -> list[tuple[str, dict]]:
        with self._lock:
            return list(self._read_all().items())


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)


class FileStateStore:
    def __init__(self, file_path: Path):
        self._store = JsonFileStore(file_path)

    async def get(self, key, options=None):
        raw = self._store.get_raw(key)
        if raw is None:
            return None
        return StateData.parse_obj(raw)

    async def set(self, key, value, options=None):
        self._store.set_raw(key, _to_plain_dict(value))

    async def delete(self, key, options=None):
        self._store.delete_raw(key)

    async def delete_by_logout_token(self, claims, options=None):
        claim_sub = getattr(claims, "sub", None) or claims.get("sub")
        claim_sid = getattr(claims, "sid", None) or claims.get("sid")
        for item_key, raw in self._store.iter_items():
            user = raw.get("user") or {}
            internal = raw.get("internal") or {}
            if user.get("sub") == claim_sub and internal.get("sid") == claim_sid:
                self._store.delete_raw(item_key)


class FileTransactionStore:
    def __init__(self, file_path: Path):
        self._store = JsonFileStore(file_path)

    async def get(self, key, options=None):
        raw = self._store.get_raw(key)
        if raw is None:
            return None
        return TransactionData.parse_obj(raw)

    async def set(self, key, value, options=None):
        self._store.set_raw(key, _to_plain_dict(value))

    async def delete(self, key, options=None):
        self._store.delete_raw(key)


state_store = FileStateStore(STATE_STORE_FILE)
transaction_store = FileTransactionStore(TRANSACTION_STORE_FILE)

auth0 = ServerClient(
    domain=get_required_env("AUTH0_DOMAIN"),
    client_id=get_required_env("AUTH0_CLIENT_ID"),
    client_secret=get_required_env("AUTH0_CLIENT_SECRET"),
    secret=get_required_env("AUTH0_SECRET"),
    redirect_uri=get_required_env("AUTH0_REDIRECT_URI"),
    state_store=state_store,
    transaction_store=transaction_store,
    authorization_params={
        "scope": "openid profile email",
        "audience": os.getenv("AUTH0_AUDIENCE", ""),
    },
)
