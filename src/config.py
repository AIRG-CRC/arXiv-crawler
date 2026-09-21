from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, ClassVar

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


def _abs(p: str | os.PathLike[str]) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


@dataclass
class Paths:
    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "data")
    metadata_file: Path | None = None

    def __post_init__(self) -> None:
        self.data_dir = _abs(self.data_dir)
        self.metadata_file = (
            self.data_dir / "raw" / "arxiv-metadata-oai-snapshot.json"
            if self.metadata_file is None
            else _abs(self.metadata_file)
        )

    # These four must agree with utils/paths.py, which is where every actual read and
    # write of an artefact resolves its location -- and which the bucket layout mirrors
    # through objectstore.local_to_object. They used to say data/processed/<kind> while
    # paths.py said data/<kind>, so the end-of-run sweep and `verify` looked for staged
    # PDFs in a directory nothing ever wrote to, and no leftover was ever cleaned up.
    # tests/test_paths.py pins the agreement.
    @property
    def md_dir(self) -> Path:
        return self.data_dir / "md"

    @property
    def tables_dir(self) -> Path:
        return self.data_dir / "tables"

    @property
    def meta_dir(self) -> Path:
        return self.data_dir / "meta"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    @property
    def eda_dir(self) -> Path:
        return self.data_dir / "processed" / "eda"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def checkpoints_dir(self) -> Path:
        return self.data_dir / "checkpoints"

    @property
    def manifest_db(self) -> Path:
        return self.data_dir / "manifest.db"

    @property
    def sync_state(self) -> Path:
        """Last bucket sync, for `status`. Deliberately not under checkpoints/: the
        `checkpoint --clear` command globs every *.json in that directory."""
        return self.data_dir / "sync-state.json"

    def ensure(self) -> None:
        """Create every output directory"""
        for d in (
            self.data_dir, self.md_dir, self.tables_dir, self.meta_dir,
            self.tmp_dir, self.eda_dir, self.logs_dir, self.checkpoints_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class Scope:
    """Defines the limitation of the papers to be processed"""
    categories: list[str] = field(default_factory=list)
    primary_only: bool = False
    date_from: str | None = None
    date_to: str | None = None
    max_papers: int | None = None


@dataclass
class Crawl:
    """Defines the crawling status"""
    base_url: str = "https://export.arxiv.org"
    contact: str = "ai@crc.calvin.ac.id"
    rate_per_sec: float = 1.0
    burst: int = 4
    workers: int = 1
    timeout: int = 60
    max_attempts: int = 5
    chunk_size: int = 65536
    # --- throttle cooldown ---
    # arXiv answers 406 (sometimes 403) when it decides a client is asking for too much.
    # It is not a fact about the paper, so it must not be handled as a per-paper error:
    # seeing one of these statuses pauses EVERY download for `cooldown_seconds`, doubling
    # each round the block is still there, up to `cooldown_max_seconds`. The paper is then
    # retried without spending an attempt. After `cooldown_max_rounds` fruitless rounds the
    # run ends with exit code 75. Set `cooldown_seconds: 0` to restore the old behaviour.
    cooldown_statuses: list[int] = field(default_factory=lambda: [403, 406])
    cooldown_seconds: int = 3600
    cooldown_max_seconds: int = 21600
    cooldown_escalate: bool = True
    cooldown_max_rounds: int = 4


@dataclass
class Convert:
    """Defines the conversion status"""
    converter: str = "docling"
    workers: int | None = 1
    timeout: int = 120
    max_pages: int = 300
    table_strategy: str = "lines_strict"
    table_fallback_strategy: str | None = None
    min_chars_per_page: int = 100
    detect_pseudocode: bool = True
    preserve_equations: bool = True
    max_table_columns: int = 25
    # Model backends only (docling). "auto" lets the library choose; naming the device
    # makes a silent fall back to CPU -- which costs an order of magnitude -- visible.
    device: str = "auto"
    # CPU threads *per worker*. Left unset it is derived from the core count so that
    # workers x threads does not oversubscribe: docling defaults to 4 threads each, so
    # 6 workers on an 8-core box would ask for 24.
    num_threads: int | None = None
    # docling's text-extraction backend. "docling_parse" is docling's own default and
    # can deadlock past any timeout; see DoclingConverter for the measurement.
    pdf_backend: str = "pypdfium"
    # Tried when `converter` fails or times out on a paper. null disables it, and the
    # paper is then recorded as failed_convert exactly as before.
    fallback_converter: str | None = "pymupdf"
    # Recycle a conversion worker after this many papers, so any growth the per-paper
    # heap release cannot reclaim is bounded by construction. Costs one model reload.
    # null keeps workers for the whole run.
    max_tasks_per_child: int | None = 100
    # Stop dispatching new papers while free memory is below this. A fraction of total
    # RAM (0.12), an absolute size ("4GB"), or null to disable the guard.
    memory_floor: float | str | None = 0.12

    def __post_init__(self) -> None:
        if not self.workers:
            self.workers = os.cpu_count() or 4
        if not self.num_threads:
            self.num_threads = max(1, (os.cpu_count() or 4) // max(self.workers, 1))


@dataclass
class Retry:
    """What `run` does with papers a previous run failed on.

    `on_start` makes every run open with a pass over the retryable failures before it
    touches fresh work, so a failure caused by something transient -- a missing optional
    dependency, a truncated download, a converter since fixed -- heals on the next run
    instead of waiting for someone to remember `retry`.
    """
    on_start: bool = True
    # Lifetime cap on how many times a paper is ever handed out. `null` means no cap:
    # every failed paper is retried on every run, which is what you want while the
    # failures are still being diagnosed, at the cost of re-downloading hopeless ones.
    max_attempts: int | None = 4
    # Retry a paper *within the same run* rather than leaving it for the next one.
    # `in_run_attempts` is how many extra tries it gets before the run gives up on it;
    # `max_attempts` still caps the lifetime total, so this cannot loop.
    in_run: bool = True
    in_run_attempts: int = 1


@dataclass
class Minio:
    """Where converted papers go when object storage is the target.

    `access_key` and `secret_key` are deliberately left null: this file is tracked in
    git, so credentials belong in MINIO_ACCESS_KEY / MINIO_SECRET_KEY instead. They are
    accepted here only so a throwaway local instance is not a nuisance to point at.
    """
    endpoint: str = "10.3.18.40:9000"
    bucket: str = "airg"
    prefix: str = "arxiv"          # everything lands under this key prefix
    secure: bool = False           # the endpoint above is plain HTTP
    access_key: str | None = None
    secret_key: str | None = None


@dataclass
class Sync:
    """Reconciling the local manifest against the shared bucket before a run.

    The three per-device settings -- `device`, `devices`, `device_index` -- can be set here
    or in the environment (ARXIV_CRAWLER_DEVICE / _DEVICES / _DEVICE_INDEX), with the
    environment winning, and a `--devices` / `--device-index` flag winning over both.

    Setting them here is the fixed, per-machine option, at one cost worth knowing: this file
    is tracked in git, so a pull, a merge or a checkout can carry one machine's
    `device_index` onto another. Two devices on the same index crawl the same slice and
    nothing crawls the rest. The startup cross-check against each device's bucket marker is
    what catches that, and it warns rather than refuses -- so read the first few lines of a
    run after you change these.
    """
    device: str | None = None
    # How many machines share this corpus, and which slice this one takes (0-based).
    # null/1 means "this machine does all of it", which is the single-device default.
    devices: int | None = None
    device_index: int | None = None
    # Require a meta object as well as an md object before believing a paper is finished.
    # The upload order is md -> tables -> meta, so a device killed between the first and
    # the last leaves an md with no meta, and nobody would ever produce it.
    require_meta: bool = True
    on_start: bool = True          # sync automatically at the start of `run-minio`
    marker: bool = True            # publish <prefix>/_state/sync/<device>.json


@dataclass
class Postgres:
    """Defines the postgres ingestion status"""
    dsn: str | None = None
    table: str = "papers"


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    scope: Scope = field(default_factory=Scope)
    crawl: Crawl = field(default_factory=Crawl)
    convert: Convert = field(default_factory=Convert)
    retry: Retry = field(default_factory=Retry)
    minio: Minio = field(default_factory=Minio)
    sync: Sync = field(default_factory=Sync)
    postgres: Postgres = field(default_factory=Postgres)

    SECTIONS: ClassVar[dict[str, type]] = {
        "paths": Paths, "scope": Scope, "crawl": Crawl,
        "convert": Convert, "retry": Retry, "minio": Minio, "sync": Sync,
        "postgres": Postgres,
    }

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "Config":
        cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
        raw: dict[str, Any] = {}
        
        if cfg_path.exists():
            raw = yaml.safe_load(cfg_path.read_text()) or {}

        kwargs: dict[str, Any] = {}
        for name, section_cls in cls.SECTIONS.items():
            section = raw.get(name) or {}
            if not isinstance(section, dict):
                raise ValueError(f"config section '{name}' must be a mapping")
            
            unknown = set(section) - {sf.name for sf in fields(section_cls)}

            if unknown:
                raise ValueError(
                    f"unknown key(s) in config section '{name}': {sorted(unknown)}"
                )
            
            kwargs[name] = section_cls(**section)
        return cls(**kwargs)

    def override(self, **flat: Any) -> "Config":
        by_field: dict[str, list[Any]] = {}
        for sec in fields(self):
            obj = getattr(self, sec.name)
            for f in fields(obj):
                by_field.setdefault(f.name, []).append(obj)

        for key, value in flat.items():
            if value is None:
                continue

            sec_name, _, field_name = key.partition("_")

            target = getattr(self, sec_name, None)

            if is_dataclass(target) and any(f.name == field_name for f in fields(target)):
                setattr(target, field_name, value)
                continue

            owners = by_field.get(key, [])

            if len(owners) == 1:
                setattr(owners[0], key, value)
            elif not owners:
                raise KeyError(f"no config field matches CLI override '{key}'")
            else:
                raise KeyError(f"CLI override '{key}' is ambiguous; qualify it as section_field")

        self.paths.__post_init__()
        self.convert.__post_init__()
        return self
