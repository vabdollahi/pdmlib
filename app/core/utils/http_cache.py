"""
Lightweight local HTTP response byte cache with TTL.

Used to cache API downloads (e.g., CAISO ZIP files) for short periods during
development to reduce network calls and rate limit hits.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.utils.logging import get_logger

logger = get_logger("http_cache")


class SimpleHTTPCache:
    """
    Simple file-based byte cache with TTL. Stores entries under a base directory
    using SHA-256 keys. Each entry has a .bin (data) and .meta.json (timestamp).

    Notes:
    - This cache is local-only and not meant for production durability.
    - TTL of 0 or negative disables cache reads (always miss) but still writes.
    """

    def __init__(
        self,
        base_dir: str | Path = ".cache/http",
        ttl_seconds: int = 600,
        namespace: str = "default",
    ) -> None:
        self.base_dir = Path(base_dir) / namespace
        self.ttl_seconds = ttl_seconds
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _paths_for_key(self, key: str) -> tuple[Path, Path]:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        subdir = self.base_dir / h[:2] / h[2:4]
        subdir.mkdir(parents=True, exist_ok=True)
        data_path = subdir / f"{h}.bin"
        meta_path = subdir / f"{h}.meta.json"
        return data_path, meta_path

    def get(self, key: str) -> Optional[bytes]:
        try:
            data_path, meta_path = self._paths_for_key(key)
            if not data_path.exists() or not meta_path.exists():
                return None

            # TTL check
            if self.ttl_seconds is not None and self.ttl_seconds > 0:
                with meta_path.open("r", encoding="utf-8") as mf:
                    meta = json.load(mf)
                ts = float(meta.get("ts", 0))
                if time.time() - ts > self.ttl_seconds:
                    return None

            return data_path.read_bytes()
        except Exception as e:
            logger.warning(f"Cache read error (key={key[:16]}...): {e}")
            return None

    def set(self, key: str, data: bytes) -> None:
        try:
            data_path, meta_path = self._paths_for_key(key)
            data_path.write_bytes(data)
            with meta_path.open("w", encoding="utf-8") as mf:
                json.dump({"ts": time.time()}, mf)
        except Exception as e:
            logger.warning(f"Cache write error (key={key[:16]}...): {e}")

    @staticmethod
    def canonical_key(endpoint: str, params: Optional[Dict[str, Any]]) -> str:
        # Deterministic key from endpoint and sorted params
        params = params or {}
        items = sorted((str(k), str(v)) for k, v in params.items())
        return f"{endpoint}?" + "&".join([f"{k}={v}" for k, v in items])
