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

    @property
    def md_dir(self) -> Path:
        return self.data_dir / "processed" / "md"

    @property
    def tables_dir(self) -> Path:
        return self.data_dir / "processed" / "tables"

    @property
    def meta_dir(self) -> Path:
        return self.data_dir / "processed" / "meta"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "processed" / "tmp"

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
    workers: int = 4
    timeout: int = 60
    max_attempts: int = 4
    chunk_size: int = 65536


@dataclass
class Convert:
    """Defines the conversion status"""
    converter: str = "pymupdf"
    workers: int | None = 8
    timeout: int = 120
    max_pages: int = 300
    table_strategy: str = "lines_strict"
    table_fallback_strategy: str | None = None
    min_chars_per_page: int = 100
    detect_pseudocode: bool = True
    preserve_equations: bool = True
    max_table_columns: int = 25

    def __post_init__(self) -> None:
        if not self.workers:
            self.workers = os.cpu_count() or 4


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
    postgres: Postgres = field(default_factory=Postgres)

    SECTIONS: ClassVar[dict[str, type]] = {
        "paths": Paths, "scope": Scope, "crawl": Crawl,
        "convert": Convert, "postgres": Postgres,
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
