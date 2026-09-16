"""Move a corpus that is already on disk into the bucket.

`dump` exists because 4.6 GB of markdown was converted before object storage was wired
in. It walks ``md/``, ``tables/`` and ``meta/``, uploads each file under the same
relative name, and removes the local copy once the upload has returned.

Resumable by construction: the local file is the record of what still needs sending, so
an interrupted dump is finished by running it again. `--keep-local` turns it into a copy
instead of a move, and `--skip-existing` avoids re-sending objects already in the bucket
at the cost of a HEAD request per file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .objectstore import KINDS, MinioStore, local_to_object

log = logging.getLogger(__name__)


@dataclass
class DumpReport:
    uploaded: int = 0
    skipped: int = 0
    failed: int = 0
    bytes_sent: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def considered(self) -> int:
        return self.uploaded + self.skipped + self.failed


def iter_artefacts(data_dir: Path) -> Iterator[Path]:
    """Every converted file on disk, in a stable order.

    Only the three output directories are walked. `tmp/` holds staged PDFs, `logs/` and
    `checkpoints/` are run bookkeeping, and none of them belong in the corpus bucket.
    """
    for kind in KINDS:
        root = data_dir / kind
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.name.startswith("."):
                yield path


def count_artefacts(data_dir: Path) -> int:
    return sum(1 for _ in iter_artefacts(data_dir))


def dump_to_bucket(
    store: MinioStore,
    data_dir: Path,
    *,
    keep_local: bool = False,
    skip_existing: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    progress: Any = None,
) -> DumpReport:
    """Upload everything under `data_dir`, removing local copies unless `keep_local`."""
    report = DumpReport()
    prefix = store.settings.prefix

    for path in iter_artefacts(data_dir):
        if limit is not None and report.considered >= limit:
            break
        name = local_to_object(path, data_dir, prefix=prefix)
        if name is None:                       # not one of ours; leave it alone
            continue

        if dry_run:
            report.uploaded += 1
            report.bytes_sent += path.stat().st_size
            log.info("would upload %s -> %s", path, name)
            if progress:
                progress.update(1)
            continue

        try:
            if skip_existing and store.exists(name):
                report.skipped += 1
            else:
                report.bytes_sent += store.put_file(path, name)
                report.uploaded += 1
            # Only ever after a successful put (or a confirmed existing object): the
            # local file is the only copy until the bucket has one.
            if not keep_local:
                path.unlink(missing_ok=True)
        except Exception as exc:               # one bad object must not end the dump
            report.failed += 1
            message = f"{path.name}: {type(exc).__name__}: {exc}"
            report.errors.append(message)
            log.error("upload failed for %s -> %s", path, name, exc_info=exc)
        if progress:
            progress.update(1)

    return report


def prune_empty_dirs(data_dir: Path) -> int:
    """Remove directories left behind by a move. Shards are per-month, so a full dump
    strands thousands of empty ones."""
    removed = 0
    for kind in KINDS:
        root = data_dir / kind
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
                removed += 1
    return removed
