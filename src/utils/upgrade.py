"""Finding the papers an earlier run converted with the fallback backend.

From this version on the manifest records which backend produced each paper, and the run
re-converts the fallback ones by itself. Manifests written before that have no such record,
but the information was never lost -- it is in two places:

  * ``crawler.log``: every fallback conversion logged ``<id> converted by fallback <name>``.
    Cheap to read, and scanned automatically at the start of a run.
  * each paper's markdown front matter in the bucket: ``converter: <name>``. Authoritative,
    but one small ranged GET per paper, so it only runs when asked for (`find-fallbacks
    --from-bucket`).

Both end in `Manifest.record_converters`, which only fills rows that are `done` and have no
converter yet -- so neither can overwrite what a run recorded first-hand.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from .state import DONE, Manifest

log = logging.getLogger(__name__)

_LOG_LINE = re.compile(r"(\S+) converted by fallback '?([\w.-]+)'? after:")
_FRONT_MATTER_CONVERTER = re.compile(rb"^converter:\s*['\"]?([\w.-]+)", re.MULTILINE)

# Enough for the front matter of any paper: the long field is the author list, and a
# collaboration paper with thousands of authors still fits comfortably.
FRONT_MATTER_BYTES = 65536
RECORD_BATCH = 500


def fallbacks_in_log(lines: Iterable[str]) -> dict[str, str]:
    """`{arxiv_id: backend}` for every fallback conversion the log mentions."""
    found: dict[str, str] = {}
    for line in lines:
        if "converted by fallback" not in line:
            continue
        match = _LOG_LINE.search(line)
        if match:
            found[match.group(1)] = match.group(2)
    return found


def scan_log(manifest: Manifest, log_path: Path, state_path: Path) -> int:
    """Record fallbacks from the part of `log_path` not read before. Returns rows filled.

    The offset reached is kept in `state_path`, so the log is read once rather than on
    every start. A log that has shrunk was rotated or deleted, and is read from the top.
    """
    try:
        size = log_path.stat().st_size
    except OSError:
        return 0
    offset = 0
    try:
        offset = int(json.loads(state_path.read_text(encoding="utf-8")).get("offset", 0))
    except (OSError, ValueError, AttributeError):
        pass
    if offset > size:
        offset = 0
    if offset == size:
        return 0
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        fh.seek(offset)
        found = fallbacks_in_log(fh)
        offset = fh.tell()
    filled = manifest.record_converters(found.items()) if found else 0
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"offset": offset}), encoding="utf-8")
    except OSError:
        log.warning("could not save the log scan offset to %s", state_path)
    if filled:
        log.info("log scan: %d paper(s) found to have been converted by a fallback", filled)
    return filled


def converter_in_front_matter(head: bytes) -> str | None:
    """The `converter:` value from the start of a stored markdown file, if it has one."""
    if not head.startswith(b"---"):
        return None
    end = head.find(b"\n---", 3)
    block = head[: end if end > 0 else len(head)]
    match = _FRONT_MATTER_CONVERTER.search(block)
    return match.group(1).decode("ascii", "replace") if match else None


def unknown_converter_ids(manifest: Manifest) -> list[str]:
    """Finished papers converted *here* whose backend the manifest does not know.

    `n_chars` is only ever written by this device's own conversion; a row marked done from
    the bucket has none. That keeps the scan to this device's papers, which is what makes
    the later upgrade a one-device job per paper.
    """
    return [r[0] for r in manifest.conn.execute(
        "SELECT arxiv_id FROM papers "
        "WHERE status = ? AND converter IS NULL AND n_chars IS NOT NULL", (DONE,)
    )]


def _batched(pairs: Iterator[tuple[str, str]], size: int) -> Iterator[list[tuple[str, str]]]:
    batch: list[tuple[str, str]] = []
    for pair in pairs:
        batch.append(pair)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def scan_bucket(
    manifest: Manifest,
    store: Any,
    *,
    limit: int | None = None,
    progress: Callable[[int], None] | None = None,
) -> dict[str, int]:
    """Read each unknown paper's front matter from the bucket and record its backend.

    Committed in batches, so an interrupted scan keeps what it learnt and the next one
    starts from what is still unknown.
    """
    ids = unknown_converter_ids(manifest)
    if limit:
        ids = ids[:limit]
    counts: dict[str, int] = {"examined": 0, "unreadable": 0}

    def _read() -> Iterator[tuple[str, str]]:
        for arxiv_id in ids:
            counts["examined"] += 1
            if progress:
                progress(1)
            try:
                head = store.get_head(store.name_for("md", arxiv_id), FRONT_MATTER_BYTES)
            except Exception as exc:        # noqa: BLE001 - missing object, network, ...
                counts["unreadable"] += 1
                log.debug("front matter of %s unreadable: %s", arxiv_id, exc)
                continue
            name = converter_in_front_matter(head)
            if name is None:
                counts["unreadable"] += 1
                continue
            counts[name] = counts.get(name, 0) + 1
            yield arxiv_id, name

    for batch in _batched(_read(), RECORD_BATCH):
        manifest.record_converters(batch)
    return counts
