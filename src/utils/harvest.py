"""Extend the manifest past the Kaggle snapshot with arXiv's live metadata.

The Kaggle dump is a frozen copy of arXiv's metadata as of the day it was downloaded. The
snapshot is loaded once with `prepare`, and every later run asks arXiv for what changed since
the manifest's high-water mark. Two sources can answer (`update.source`):

* ``api`` (default) -- the search API at export.arxiv.org/api/query, sorted by last update,
  newest first, read back page by page until the entries are older than the watermark.
  This is the one that works: in November 2026 it answered in under a second, while
  uncached OAI `ListRecords` requests hit arXiv's CDN timeout (~185 s, then 503) every time.
  Two quirks shape it: a `lastUpdatedDate:[...]` filter is silently rewritten into
  `submittedDate`, so it would miss every new *version* -- hence sorting instead of
  filtering -- and one query can page through at most 30,000 results.
* ``oai`` -- OAI-PMH `ListRecords` in the `arXivRaw` format, the format the Kaggle dump is
  generated from. Kept for when arXiv's OAI service is healthy again.

Either way each record is rewritten into the *Kaggle record shape* before anything else
sees it, so `parse_record`, `matches_scope` and `to_row` in `prepare_data` serve every source
unchanged -- a paper harvested here is indistinguishable from one prepared from the snapshot.

The watermark is the `until` date of the last harvest that completed, kept in the manifest's
`kv` table. It moves only after the whole range has been read: a harvest interrupted half way
starts again from the old watermark, and the inserts it repeats are no-ops.
"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from email.utils import format_datetime
from typing import Any, Iterator

import requests

from .logging_setup import console
from .prepare_data import matches_scope, parse_record, to_row
from .state import Manifest

log = logging.getLogger(__name__)

WATERMARK_KEY = "oai_watermark"

NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "raw": "http://arxiv.org/OAI/arXivRaw/",
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "os": "http://a9.com/-/spec/opensearch/1.1/",
}


class HarvestError(RuntimeError):
    pass


# --- the feeds ----------------------------------------------------------------------------
class _FeedClient:
    """HTTP plumbing shared by both sources: identification, retries, back-off.

    Every client exposes `records(since, until)`, yielding Kaggle-shaped dicts, and
    `complete`, which is False when the source could not reach back as far as `since`.
    """

    complete = True

    # A 503 that took this long to arrive is a gateway giving up on a slow page, not OAI
    # flow control -- its `Retry-After: 1` is meaningless, and obeying it just queues the
    # same doomed request again. Measured on oaipmh.arxiv.org: ~185 s, then 503.
    SLOW_503_SECONDS = 30

    def __init__(self, ucfg: Any, contact: str, *, session: requests.Session | None = None,
                 sleep=time.sleep, clock=time.monotonic):
        from .crawler import __version__

        self.cfg = ucfg
        self.sleep = sleep
        self.clock = clock
        self.session = session or requests.Session()
        self.session.headers.setdefault(
            "User-Agent", f"arxiv-crawler/{__version__} (+{contact})")

    def _backoff(self, attempt: int) -> int:
        return min(60 * 2 ** (attempt - 1), int(getattr(self.cfg, "max_backoff", 900)))

    def _get(self, url: str, params: dict[str, str]) -> ET.Element:
        attempts = max(1, int(self.cfg.max_attempts))
        for attempt in range(1, attempts + 1):
            last = attempt == attempts
            started = self.clock()
            try:
                resp = self.session.get(url, params=params, timeout=self.cfg.timeout)
            except requests.RequestException as exc:
                if last:
                    raise HarvestError(f"OAI request to {url} failed: {exc}") from exc
                wait = self._backoff(attempt)
                log.warning("OAI request failed (%s); retrying in %ds", exc, wait)
                self.sleep(wait)
                continue
            elapsed = self.clock() - started
            if resp.status_code in (429, 503):
                if last:
                    break
                if elapsed >= self.SLOW_503_SECONDS:
                    wait = self._backoff(attempt)
                    log.warning("OAI server timed out after %ds building the page (503); "
                                "retrying in %ds", elapsed, wait)
                else:
                    # Genuine flow control (503) or rate limiting (429): the server says
                    # when to come back; a 429 without a usable hint waits a full backoff.
                    default = self._backoff(attempt) if resp.status_code == 429 else 30 * attempt
                    wait = _retry_after(resp.headers.get("Retry-After"), default=default)
                    if resp.status_code == 429:
                        wait = max(wait, self._backoff(attempt))
                    log.info("OAI server asked to retry in %ds", wait)
                self.sleep(wait)
                continue
            if resp.status_code >= 400:
                if last or resp.status_code < 500:
                    raise HarvestError(f"OAI request to {url} returned HTTP {resp.status_code}")
                self.sleep(self._backoff(attempt))
                continue
            try:
                return ET.fromstring(resp.content)
            except ET.ParseError as exc:
                if last:
                    raise HarvestError(f"OAI response was not valid XML: {exc}") from exc
                self.sleep(self._backoff(attempt))
        raise HarvestError(f"arXiv at {url} kept answering 503/429; giving up for now")


class OaiClient(_FeedClient):
    """A minimal OAI-PMH ListRecords client, with a second endpoint to fall back on."""

    def __init__(self, ucfg: Any, contact: str, **kwargs: Any):
        super().__init__(ucfg, contact, **kwargs)
        fallback = getattr(ucfg, "fallback_oai_url", None)
        self.urls = [ucfg.oai_url] + ([fallback] if fallback and fallback != ucfg.oai_url
                                      else [])

    def records(self, since: str, until: str | None = None) -> Iterator[dict]:
        for record in self.list_records(since, until):
            rec = to_snapshot_record(record)
            if rec is not None:
                yield rec

    def list_records(self, since: str, until: str | None = None,
                     set_spec: str | None = None) -> Iterator[ET.Element]:
        """Every record in the range, from the first endpoint that serves it.

        The fallback is tried only if an endpoint fails *before its first page*: a
        resumption token belongs to the server that issued it, and restarting a listing
        half way on another server would hand the caller its first pages twice.
        """
        for i, url in enumerate(self.urls):
            pages = self._pages(url, since, until, set_spec)
            try:
                first = next(pages)
            except StopIteration:
                return
            except HarvestError as exc:
                if i + 1 == len(self.urls):
                    raise
                log.warning("%s; trying %s instead", exc, self.urls[i + 1])
                console("arXiv OAI endpoint %s failed (%s); falling back to %s",
                        url, exc, self.urls[i + 1])
                continue
            yield from first
            for page in pages:
                yield from page
            return

    def _pages(self, url: str, since: str, until: str | None,
               set_spec: str | None) -> Iterator[list[ET.Element]]:
        params = {"verb": "ListRecords", "metadataPrefix": self.cfg.metadata_prefix,
                  "from": since}
        if until:
            params["until"] = until
        if set_spec:
            params["set"] = set_spec
        page = 0
        while True:
            root = self._get(url, params)
            err = root.find("oai:error", NS)
            if err is not None:
                if err.get("code") == "noRecordsMatch":
                    return
                raise HarvestError(f"OAI error {err.get('code')}: {(err.text or '').strip()}")
            listing = root.find("oai:ListRecords", NS)
            if listing is None:
                return
            yield listing.findall("oai:record", NS)
            page += 1
            token = listing.find("oai:resumptionToken", NS)
            if token is None or not (token.text or "").strip():
                return
            log.debug("OAI page %d done; cursor %s of %s", page,
                      token.get("cursor"), token.get("completeListSize"))
            # A resumption request carries the token and nothing else, by the protocol.
            params = {"verb": "ListRecords", "resumptionToken": token.text.strip()}
            self.sleep(self.cfg.page_delay)


class ApiClient(_FeedClient):
    """arXiv's search API, read newest-updated first until it is older than `since`."""

    # arXiv refuses to page a single query past this many results.
    MAX_RESULTS = 30_000

    def __init__(self, ucfg: Any, contact: str, *, categories: list[str] | None = None,
                 **kwargs: Any):
        super().__init__(ucfg, contact, **kwargs)
        # Narrowing the query to the scope's categories is what keeps a long gap inside the
        # 30,000 ceiling; `primary_only` is still applied by `matches_scope` afterwards.
        self.query = " OR ".join(f"cat:{c}" for c in categories) if categories else "cat:*"
        self.complete = True
        self.oldest_seen: str | None = None

    def _page(self, start: int) -> tuple[list[ET.Element], int]:
        params = {"search_query": self.query, "sortBy": "lastUpdatedDate",
                  "sortOrder": "descending", "start": str(start),
                  "max_results": str(self.cfg.api_page_size)}
        attempts = max(1, int(self.cfg.max_attempts))
        for attempt in range(1, attempts + 1):
            root = self._get(self.cfg.api_url, params)
            total = int(root.findtext("os:totalResults", default="0", namespaces=NS) or 0)
            entries = root.findall("atom:entry", NS)
            # The API occasionally answers a page in the middle of a listing with no
            # entries at all. Retrying the same page fixes it; trusting it ends the harvest
            # early and silently.
            if entries or start >= total or attempt == attempts:
                return entries, total
            log.info("arXiv API returned an empty page at %d of %d; retrying", start, total)
            self.sleep(self.cfg.page_delay * attempt)
        return [], 0

    def records(self, since: str, until: str | None = None) -> Iterator[dict]:
        start, size = 0, int(self.cfg.api_page_size)
        while True:
            if start >= self.MAX_RESULTS:
                self.complete = False       # cannot page further back than this
                return
            entries, total = self._page(start)
            if not entries:
                if start < total:
                    self.complete = False   # gave up on an empty page mid-listing
                return
            for entry in entries:
                rec = api_entry_to_snapshot_record(entry)
                if rec is None:
                    continue
                day = rec["update_date"]
                self.oldest_seen = day
                if until and day > until:
                    continue
                if day < since:
                    continue
                yield rec
            # Sorted newest first, so once a page ends before `since` nothing further back
            # can be in range. Checked per page, not per entry, so a few out-of-order
            # entries at a boundary cannot stop the walk early.
            if self.oldest_seen and self.oldest_seen < since:
                return
            start += len(entries)
            if start >= total:
                return
            self.sleep(self.cfg.page_delay)


def _retry_after(value: str | None, *, default: int) -> int:
    try:
        return max(1, int(value)) if value else default
    except ValueError:
        return default


# --- one record ---------------------------------------------------------------------------
_AND = re.compile(r",?\s+and\s+", re.IGNORECASE)
_PARENS = re.compile(r"\([^()]*\)")


def split_authors(raw: str) -> list[list[str]]:
    """`"A. Smith, B. Jones and C. Lee"` -> `[["Smith", "A.", ""], ...]`.

    Kaggle's `authors_parsed` comes from arXiv's own parser, which this does not try to
    match in every corner (collaborations, "et al", nested affiliations). It covers the
    ordinary case, which is what `parse_record` turns into "Lastname, Firstname".
    """
    text = " ".join(raw.split())
    while _PARENS.search(text):                 # affiliations, possibly nested
        text = _PARENS.sub("", text)
    parsed = []
    for name in _AND.sub(", ", text).split(","):
        name = name.strip()
        if not name:
            continue
        suffix = ""
        parts = name.split()
        if len(parts) > 1 and parts[-1].rstrip(".") in {"Jr", "Sr", "II", "III", "IV"}:
            suffix = parts.pop()
        last, first = parts[-1], " ".join(parts[:-1])
        parsed.append([last, first, suffix])
    return parsed


def to_snapshot_record(record: ET.Element) -> dict | None:
    """One OAI `<record>` as the dict a line of the Kaggle snapshot would parse to.

    Returns None for deleted records and for records without arXivRaw metadata.
    """
    header = record.find("oai:header", NS)
    if header is None or header.get("status") == "deleted":
        return None
    raw = record.find("oai:metadata/raw:arXivRaw", NS)
    if raw is None:
        return None

    def text(tag: str) -> str | None:
        node = raw.find(f"raw:{tag}", NS)
        return node.text.strip() if node is not None and node.text else None

    versions = [
        {"version": v.get("version", "v1"),
         "created": (v.findtext("raw:date", default="", namespaces=NS) or "").strip()}
        for v in raw.findall("raw:version", NS)
    ]
    authors = text("authors") or ""
    return {
        "id": text("id") or "",
        "submitter": text("submitter"),
        "authors": authors,
        "title": text("title") or "",
        "comments": text("comments"),
        "journal-ref": text("journal-ref"),
        "doi": text("doi"),
        "categories": text("categories") or "",
        "license": text("license"),
        "abstract": text("abstract") or "",
        "versions": versions,
        "update_date": header.findtext("oai:datestamp", default=None, namespaces=NS),
        "authors_parsed": split_authors(authors),
    }


_ABS_ID = re.compile(r"/abs/(?P<id>.+?)(?P<version>v\d+)?$")


def _rfc2822(iso: str | None) -> str:
    """`2026-11-05T00:06:00Z` -> the `Thu, 05 Nov 2026 00:06:00 GMT` the snapshot uses."""
    if not iso:
        return ""
    when = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return format_datetime(when.astimezone(timezone.utc), usegmt=True)


def api_entry_to_snapshot_record(entry: ET.Element) -> dict | None:
    """One search-API Atom `<entry>` in the Kaggle record shape.

    The API gives only the current version and two dates -- `published` (v1) and `updated`
    (the current version) -- not the full history. `parse_record` reads exactly those two
    ends of the list, so a two-element history is all it needs.
    """
    match = _ABS_ID.search((entry.findtext("atom:id", default="", namespaces=NS)).strip())
    if not match:
        return None
    version = match.group("version") or "v1"
    published = entry.findtext("atom:published", default="", namespaces=NS).strip()
    updated = entry.findtext("atom:updated", default="", namespaces=NS).strip() or published
    versions = [{"version": "v1", "created": _rfc2822(published)}]
    if version != "v1":
        versions.append({"version": version, "created": _rfc2822(updated)})

    primary = entry.find("arxiv:primary_category", NS)
    categories = [primary.get("term")] if primary is not None and primary.get("term") else []
    for cat in entry.findall("atom:category", NS):
        term = cat.get("term")
        if term and term not in categories:
            categories.append(term)

    names = [" ".join((a.findtext("atom:name", default="", namespaces=NS)).split())
             for a in entry.findall("atom:author", NS)]
    names = [n for n in names if n]
    parsed_authors = [p for n in names for p in split_authors(n)[:1]]
    doi = entry.findtext("arxiv:doi", default=None, namespaces=NS)
    return {
        "id": match.group("id"),
        "authors": ", ".join(names),
        "title": entry.findtext("atom:title", default="", namespaces=NS),
        "comments": entry.findtext("arxiv:comment", default=None, namespaces=NS),
        "journal-ref": entry.findtext("arxiv:journal_ref", default=None, namespaces=NS),
        "doi": doi.strip() if doi else None,
        "categories": " ".join(categories),
        "abstract": (entry.findtext("atom:summary", default="", namespaces=NS)).strip(),
        "versions": versions,
        "update_date": updated[:10],
        "authors_parsed": parsed_authors,
    }


def make_client(cfg: Any) -> _FeedClient:
    source = getattr(cfg.update, "source", "api")
    if source == "api":
        return ApiClient(cfg.update, cfg.crawl.contact, categories=list(cfg.scope.categories))
    if source == "oai":
        return OaiClient(cfg.update, cfg.crawl.contact)
    raise HarvestError(f"update.source must be 'api' or 'oai', got {source!r}")


# --- the harvest --------------------------------------------------------------------------
def resolve_since(manifest: Manifest, overlap_days: int, explicit: str | None = None) -> str:
    """Where to start: the CLI value, else the stored watermark, else the newest paper the
    snapshot holds -- each minus `overlap_days`, because OAI datestamps are per day and a
    paper changed late on the boundary day must not slip between two harvests."""
    if explicit:
        return explicit
    base = manifest.get_kv(WATERMARK_KEY) or manifest.latest_update_date()
    if not base:
        raise HarvestError(
            "the manifest is empty and no --from date was given; run `prepare` on the "
            "Kaggle snapshot first, or pass --from YYYY-MM-DD")
    start = date.fromisoformat(base[:10]) - timedelta(days=max(0, overlap_days))
    return start.isoformat()


def harvest(
    cfg: Any,
    manifest: Manifest,
    *,
    since: str | None = None,
    until: str | None = None,
    dry_run: bool = False,
    client: _FeedClient | None = None,
    batch_size: int = 2000,
) -> dict[str, Any]:
    """Pull every record changed in `[since, until]` into the manifest."""
    ucfg = cfg.update
    since = resolve_since(manifest, ucfg.overlap_days, since)
    # The watermark stored afterwards is "today" in UTC unless an explicit end was given.
    stamp = until or datetime.now(timezone.utc).date().isoformat()
    client = client or make_client(cfg)
    counts: dict[str, Any] = {"since": since, "until": stamp, "seen": 0, "matched": 0,
                              "inserted": 0, "version_bumped": 0, "requeued": 0}
    console("checking arXiv for papers changed since %s", since)

    batch = []

    def _flush() -> None:
        got = manifest.apply_updates(
            batch, recrawl=ucfg.recrawl_new_versions, dry_run=dry_run)
        for key in ("inserted", "version_bumped", "requeued"):
            counts[key] += got[key]
        batch.clear()

    # `scope` is applied on this side for every source (the API query is only narrowed to
    # the scope's categories, never trusted to be exact), so the rules match `prepare`'s.
    for rec in client.records(since, until):
        counts["seen"] += 1
        parsed = parse_record(rec)
        if not parsed["id"] or not matches_scope(parsed, cfg.scope):
            continue
        counts["matched"] += 1
        batch.append(to_row(parsed))
        if len(batch) >= batch_size:
            _flush()
    if batch:
        _flush()

    counts["complete"] = client.complete
    if not client.complete:
        # The newest papers are in; the oldest stretch of the gap is not. Keeping the old
        # watermark means the next run tries again rather than declaring the gap closed --
        # and the inserts it repeats are no-ops.
        reached = getattr(client, "oldest_seen", None) or "?"
        log.warning("arXiv would only page back to %s, not %s; the watermark stays put. "
                    "Narrow scope.categories, or refresh the Kaggle snapshot and re-run "
                    "`prepare` to cover the rest.", reached, since)
        console("⚠ arXiv's API only reaches back to %s of the %s → %s gap; the papers in "
                "between were not seen. Narrow scope.categories, or load a newer Kaggle "
                "snapshot with `prepare`.", reached, since, stamp)
    elif not dry_run:
        manifest.set_kv(WATERMARK_KEY, stamp)
    return counts
