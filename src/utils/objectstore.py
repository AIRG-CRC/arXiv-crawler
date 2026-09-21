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
from datetime import datetime
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


# What each kind's leaf looks like, so a name can be split back apart. `md` is listed
# last on purpose: `.tables.md` also ends in `.md`, so it has to be ruled out first.
_LEAF_SUFFIX = {"tables": ".tables.md", "meta": ".json", "md": ".md"}


def id_candidates_from_object_name(
    name: str, *, prefix: str = "", kind: str = "md"
) -> tuple[str, ...]:
    """The arXiv id(s) an object name could have come from; empty if it is not one.

    `safe_id` maps "/" to "_", which is injective over real ids -- modern stems contain no
    underscore, old-style stems contain exactly one, and no arXiv id contains an underscore
    -- but rather than lean on that, both readings are returned and the manifest decides.
    At most one of them can be a row, since one contains a slash and the other does not.
    """
    parts = name.rsplit("/", 3)
    if len(parts) == 4:
        _head, obj_kind, _shard, leaf = parts
    elif len(parts) == 3:
        obj_kind, _shard, leaf = parts
    else:
        return ()
    if obj_kind != kind:
        return ()
    suffix = _LEAF_SUFFIX[kind]
    if kind == "md" and leaf.endswith(_LEAF_SUFFIX["tables"]):
        return ()                       # a tables object, which also ends in ".md"
    if not leaf.endswith(suffix) or len(leaf) <= len(suffix):
        return ()
    stem = leaf[: -len(suffix)]
    if "_" not in stem:
        return (stem,)
    return (stem, stem.replace("_", "/", 1))


def id_from_object_name(name: str, *, prefix: str = "", kind: str = "md") -> str | None:
    """The one id that maps back to exactly this name, or None.

    Generate-and-verify rather than a second parser: the answer is whichever candidate
    `object_name` rebuilds into the name we were given, so the inverse cannot drift away
    from the forward map as the layout changes.
    """
    for candidate in id_candidates_from_object_name(name, prefix=prefix, kind=kind):
        if object_name(kind, candidate, prefix=prefix) == name:
            return candidate
    return None


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

    def get_bytes(self, name: str) -> bytes:
        """Download one small object. Only used for the device markers.

        The SDK hands back a urllib3 response that has to be closed *and* have its
        connection released, or the pool leaks a socket per call.
        """
        response = self.client.get_object(self.settings.bucket, name)
        try:
            return response.read()
        finally:
            for method in ("close", "release_conn"):
                closer = getattr(response, method, None)
                if closer is not None:
                    closer()

    def exists(self, name: str) -> bool:
        try:
            self.client.stat_object(self.settings.bucket, name)
            return True
        except Exception:               # the SDK raises S3Error for a missing key
            return False

    def bucket_present(self) -> bool:
        """Does the bucket exist? Unlike `ensure_bucket`, this never creates it.

        Which is what a reader wants: `ensure_bucket` would turn a typo'd bucket name into
        a new empty bucket, and the sync would then cheerfully report nothing to do.
        """
        return bool(self.client.bucket_exists(self.settings.bucket))

    def _iter_raw(
        self, prefix: str | None, *, start_after: str | None = None
    ) -> Iterator[tuple[str, int, datetime | None]]:
        for obj in self.client.list_objects(
            self.settings.bucket, prefix=prefix or None, recursive=True,
            start_after=start_after,
        ):
            yield (obj.object_name,
                   getattr(obj, "size", 0) or 0,
                   getattr(obj, "last_modified", None))

    def list_names(self, prefix: str | None = None) -> Iterator[str]:
        """Object names under an absolute prefix (default: the configured one)."""
        target = self.settings.prefix if prefix is None else prefix
        for name, _size, _modified in self._iter_raw(target):
            yield name

    def iter_objects(
        self, relative: str = "", *, start_after: str | None = None
    ) -> Iterator[tuple[str, int, datetime | None]]:
        """`(name, size, last_modified)` for everything under `relative`.

        `relative` is joined onto the configured prefix and given a trailing slash, because
        a bare "md" would also match an "mdx/" that someone adds later.

        `start_after` resumes a listing within one call. It must **never** be kept as a
        high-water mark between runs. Object names sort as `md/<shard>/<id>.md`, and
        (1) claims come back in rowid order, which has nothing to do with key order, so a
        run writes objects all over the keyspace; (2) old-style ids shard to `9901`, which
        sorts after every modern `2xxx` shard; (3) `misc` sorts after all of them. Any
        stored cursor therefore skips papers silently and permanently.
        """
        target = "/".join(
            p for p in (self.settings.prefix.strip("/"), relative.strip("/")) if p
        )
        yield from self._iter_raw(f"{target}/" if target else None, start_after=start_after)

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
