"""MinIO object storage for the converted corpus.

The bucket mirrors the local layout exactly -- ``md/<shard>/<id>.md``,
``tables/<shard>/<id>.tables.md``, ``meta/<shard>/<id>.json`` under an optional prefix --
so an object name can be derived from an arXiv id without consulting anything, and a
half-migrated corpus can be read from either side.

Writing goes local-then-upload-then-delete rather than straight to the network. The
atomic local write already exists and is the thing that makes an interrupted run safe; if
the upload fails, the file is still on disk and the next `dump` picks it up. Only a
confirmed upload removes it.

Credentials are read from ``MINIO_ACCESS_KEY`` / ``MINIO_SECRET_KEY``. `config.yaml` is
tracked in git, so it is the wrong place for a secret and the loader says so out loud.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .paths import safe_id, shard_for

log = logging.getLogger(__name__)

ACCESS_KEY_ENV = "MINIO_ACCESS_KEY"
SECRET_KEY_ENV = "MINIO_SECRET_KEY"

# The three artefacts, and how each maps to an object name.
KINDS = ("md", "tables", "meta")


class ObjectStoreError(RuntimeError):
    pass


def object_name(kind: str, arxiv_id: str, *, prefix: str = "") -> str:
    """Where one artefact lives in the bucket. Mirrors `utils.paths`."""
    stem = safe_id(arxiv_id)
    leaf = {
        "md": f"{stem}.md",
        "tables": f"{stem}.tables.md",
        "meta": f"{stem}.json",
    }[kind]
    parts = [p for p in (prefix.strip("/"), kind, shard_for(arxiv_id), leaf) if p]
    return "/".join(parts)


def local_to_object(path: Path, data_dir: Path, *, prefix: str = "") -> str | None:
    """The object name for a file already on disk, or None if it is not an artefact.

    Derived from the path rather than re-deriving from an id, so `dump` can walk the
    output directories without parsing filenames back into arXiv ids.
    """
    try:
        relative = path.relative_to(data_dir)
    except ValueError:
        return None
    if not relative.parts or relative.parts[0] not in KINDS:
        return None
    parts = [p for p in (prefix.strip("/"), *relative.parts) if p]
    return "/".join(parts)


@dataclass
class MinioSettings:
    """Everything needed to reach the bucket, with the secrets kept out of the file."""

    endpoint: str
    bucket: str
    prefix: str = ""
    secure: bool = False
    access_key: str | None = None
    secret_key: str | None = None

    @classmethod
    def from_config(cls, cfg: Any) -> "MinioSettings":
        access = os.environ.get(ACCESS_KEY_ENV) or getattr(cfg, "access_key", None)
        secret = os.environ.get(SECRET_KEY_ENV) or getattr(cfg, "secret_key", None)
        if getattr(cfg, "access_key", None) or getattr(cfg, "secret_key", None):
            log.warning(
                "minio credentials found in config.yaml; that file is tracked in git. "
                "Prefer %s / %s in the environment.", ACCESS_KEY_ENV, SECRET_KEY_ENV
            )
        return cls(
            endpoint=getattr(cfg, "endpoint", "") or "",
            bucket=getattr(cfg, "bucket", "") or "",
            prefix=getattr(cfg, "prefix", "") or "",
            secure=bool(getattr(cfg, "secure", False)),
            access_key=access,
            secret_key=secret,
        )

    def validate(self) -> None:
        missing = [name for name, value in (
            ("minio.endpoint", self.endpoint),
            ("minio.bucket", self.bucket),
            (ACCESS_KEY_ENV, self.access_key),
            (SECRET_KEY_ENV, self.secret_key),
        ) if not value]
        if missing:
            raise ObjectStoreError(
                "minio is not configured; missing " + ", ".join(missing)
                + f". Set the keys with:  export {ACCESS_KEY_ENV}=... {SECRET_KEY_ENV}=..."
            )


class MinioStore:
    """A thin, lazily connected wrapper over the MinIO SDK.

    Lazy because a worker process should not pay for a connection it may never use, and
    because `minio` is an optional dependency: nothing here is imported unless a MinIO
    command is actually run.
    """

    def __init__(self, settings: MinioSettings, *, client: Any = None):
        self.settings = settings
        self._client = client          # injectable, which is what makes this testable
        self._bucket_checked = client is not None

    @property
    def client(self) -> Any:
        if self._client is None:
            self.settings.validate()
            try:
                from minio import Minio
            except ImportError as exc:
                raise ObjectStoreError(
                    "the minio package is not installed. Run:  pip install minio"
                ) from exc
            self._client = Minio(
                self.settings.endpoint,
                access_key=self.settings.access_key,
                secret_key=self.settings.secret_key,
                secure=self.settings.secure,
            )
        return self._client

    def ensure_bucket(self) -> bool:
        """Create the bucket if it is absent. Returns True if it had to be created."""
        if self._bucket_checked:
            return False
        created = False
        if not self.client.bucket_exists(self.settings.bucket):
            self.client.make_bucket(self.settings.bucket)
            created = True
            log.info("created bucket %s", self.settings.bucket)
        self._bucket_checked = True
        return created

    def name_for(self, kind: str, arxiv_id: str) -> str:
        return object_name(kind, arxiv_id, prefix=self.settings.prefix)

    def put_file(self, local_path: Path, name: str) -> int:
        """Upload one file. Returns the byte count sent."""
        self.ensure_bucket()
        self.client.fput_object(self.settings.bucket, name, str(local_path))
        return local_path.stat().st_size

    def put_bytes(self, name: str, payload: bytes, content_type: str = "text/markdown") -> int:
        import io

        self.ensure_bucket()
        self.client.put_object(
            self.settings.bucket, name, io.BytesIO(payload), len(payload),
            content_type=content_type,
        )
        return len(payload)

    def exists(self, name: str) -> bool:
        try:
            self.client.stat_object(self.settings.bucket, name)
            return True
        except Exception:               # the SDK raises S3Error for a missing key
            return False

    def list_names(self, prefix: str | None = None) -> Iterator[str]:
        target = self.settings.prefix if prefix is None else prefix
        for obj in self.client.list_objects(
            self.settings.bucket, prefix=target or None, recursive=True
        ):
            yield obj.object_name

    def describe(self) -> str:
        scheme = "https" if self.settings.secure else "http"
        where = f"{scheme}://{self.settings.endpoint}/{self.settings.bucket}"
        return where + (f"/{self.settings.prefix.strip('/')}" if self.settings.prefix else "")


def upload_and_unlink(store: MinioStore, local_path: Path, name: str) -> int:
    """Upload one artefact and remove the local copy once the upload has returned.

    The unlink is deliberately *after* a successful `put`: an upload that throws leaves
    the file on disk, where the next `dump` will find it. Losing the only copy because
    the network blinked is the one failure this ordering rules out.
    """
    size = store.put_file(local_path, name)
    local_path.unlink(missing_ok=True)
    return size
