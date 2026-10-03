"""Upload failures are not converter failures, and fallback papers get re-converted.

Two things that used to be tangled together. A MinIO `IncompleteBody` on docling's output
was caught by the same handler as a docling crash, so the paper was converted again by
pymupdf and that plainer copy stored instead. Now the upload is retried on a fresh
connection, never triggers the fallback, and any paper that did end up with fallback output
is re-converted by the primary backend on a later run -- over the same objects.
"""

from __future__ import annotations

import sqlite3
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from src.config import Config, Convert, Paths
from src.utils import converter as CV
from src.utils import crawler as C
from src.utils import objectstore as OS
from src.utils.converter import (
    REGISTRY, BaseConverter, ConversionResult, TableBlock, convert_and_write,
)
from src.utils.objectstore import MinioSettings, MinioStore, UploadError, upload_and_unlink
from src.utils.state import (
    DONE, FAILED_CONVERT, SCHEMA, Manifest, ManifestWriter, PaperRow, TaskResult,
)
from src.utils.upgrade import (
    converter_in_front_matter, fallbacks_in_log, scan_bucket, scan_log,
)
from tests.test_objectstore import FakeClient


# --- uploads --------------------------------------------------------------------------
class _Pool:
    def __init__(self):
        self.cleared = 0

    def clear(self):
        self.cleared += 1


class _FlakyClient(FakeClient):
    """Rejects the first `failures` puts, the way a stale connection does."""

    def __init__(self, failures):
        super().__init__()
        self.failures = failures
        self._http = _Pool()

    def fput_object(self, bucket, name, path):
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("S3 operation failed; code: IncompleteBody")
        super().fput_object(bucket, name, path)


def _store(client, attempts=4):
    settings = MinioSettings(endpoint="host:9000", bucket="airg", prefix="arxiv",
                             access_key="k", secret_key="s", upload_attempts=attempts)
    return MinioStore(settings, client=client)


def test_an_upload_is_retried_on_a_fresh_connection(tmp_path):
    client = _FlakyClient(failures=2)
    local = tmp_path / "paper.md"
    local.write_text("body")
    waits: list[float] = []

    upload_and_unlink(_store(client), local, "arxiv/md/2301/paper.md", sleep=waits.append)

    assert client.objects["arxiv/md/2301/paper.md"] == b"body"
    assert not local.exists()
    assert client._http.cleared == 2            # one reconnect per failure
    assert waits == [1.0, 2.0]


def test_an_upload_that_never_succeeds_raises_upload_error_and_keeps_the_file(tmp_path):
    client = _FlakyClient(failures=99)
    local = tmp_path / "paper.md"
    local.write_text("body")

    with pytest.raises(UploadError) as raised:
        upload_and_unlink(_store(client, attempts=3), local, "arxiv/md/2301/paper.md",
                          sleep=lambda s: None)

    assert "3 attempt(s)" in str(raised.value) and "IncompleteBody" in str(raised.value)
    assert local.read_text() == "body"


def test_a_connection_left_idle_is_not_reused(tmp_path, monkeypatch):
    client = _FlakyClient(failures=0)
    store = _store(client)
    local = tmp_path / "paper.md"
    local.write_text("body")

    store.put_file(local, "arxiv/md/2301/a.md")
    assert client._http.cleared == 0            # used a moment ago: keep it
    store._last_used -= OS.IDLE_RECONNECT_SECONDS + 1
    store.put_file(local, "arxiv/md/2301/b.md")
    assert client._http.cleared == 1


# --- convert_and_write ----------------------------------------------------------------
class _Primary(BaseConverter):
    name = "_primary"
    calls = 0
    tables: list = []

    def convert(self, pdf_path):
        type(self).calls += 1
        return ConversionResult(body_markdown="Primary.", n_pages=2, n_chars=8,
                                tables=list(self.tables))


class _Plain(BaseConverter):
    name = "_plain"
    calls = 0

    def convert(self, pdf_path):
        type(self).calls += 1
        return ConversionResult(body_markdown="Plain.", n_pages=2, n_chars=6)


class _Broken(BaseConverter):
    name = "_broken"

    def convert(self, pdf_path):
        raise RuntimeError("primary exploded")


@pytest.fixture()
def backends():
    _Primary.calls = _Plain.calls = 0
    _Primary.tables = []
    for cls in (_Primary, _Plain, _Broken):
        REGISTRY[cls.name] = cls
    yield
    for cls in (_Primary, _Plain, _Broken):
        del REGISTRY[cls.name]
        CV._WORKER_CACHE.pop(cls.name, None)


def _row(arxiv_id="2301.00001"):
    return PaperRow(arxiv_id=arxiv_id, version="v1", shard="2301", title="T",
                    authors='["A, B"]', categories="cs.LG", primary_category="cs.LG")


def _pdf():
    tmp = Path(tempfile.mkdtemp())
    pdf = tmp / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    return tmp, pdf


def test_a_failed_upload_does_not_send_the_paper_to_the_fallback(backends, monkeypatch):
    def refuse(kind, path):
        raise UploadError("arxiv/md/2301/2301.00001.md not stored after 4 attempt(s)")

    monkeypatch.setattr(CV, "_worker_uploader", lambda minio, arxiv_id: refuse)
    tmp, pdf = _pdf()

    result = convert_and_write(
        _row(), pdf, tmp, Convert(converter="_primary", fallback_converter="_plain", timeout=5),
        base_url="https://x", minio={"endpoint": "h"},
    )

    assert result.status == FAILED_CONVERT
    assert result.error.startswith("upload:")
    assert _Plain.calls == 0                    # the conversion was fine; leave it alone
    # The primary's output is still on disk for `dump`.
    assert "converter: _primary" in (tmp / "md" / "2301" / "2301.00001.md").read_text()


def test_an_upgrade_tries_only_the_primary_and_writes_nothing_when_it_fails(backends):
    tmp, pdf = _pdf()
    result = convert_and_write(
        _row(), pdf, tmp, Convert(converter="_broken", fallback_converter="_plain", timeout=5),
        base_url="https://x", upgrade=True,
    )
    assert result.status == FAILED_CONVERT
    assert _Plain.calls == 0
    assert not (tmp / "md").exists()


def test_an_upgrade_without_tables_removes_the_stale_tables_object(backends, monkeypatch):
    removed: list[str] = []
    monkeypatch.setattr(CV, "_worker_uploader", lambda minio, arxiv_id: lambda kind, path: None)
    monkeypatch.setattr(CV, "_worker_remover", lambda minio, arxiv_id: removed.append)
    tmp, pdf = _pdf()

    result = convert_and_write(
        _row(), pdf, tmp, Convert(converter="_primary", fallback_converter="_plain", timeout=5),
        base_url="https://x", minio={"endpoint": "h"}, upgrade=True,
    )

    assert result.status == DONE and result.converter == "_primary"
    assert removed == ["tables"]


def test_an_upgrade_with_tables_keeps_them(backends, monkeypatch):
    removed: list[str] = []
    _Primary.tables = [TableBlock(index=1, page=1, n_rows=1, n_cols=1,
                                 markdown="| a |\n|---|\n| 1 |")]
    monkeypatch.setattr(CV, "_worker_uploader", lambda minio, arxiv_id: lambda kind, path: None)
    monkeypatch.setattr(CV, "_worker_remover", lambda minio, arxiv_id: removed.append)
    tmp, pdf = _pdf()

    convert_and_write(
        _row(), pdf, tmp, Convert(converter="_primary", fallback_converter="_plain", timeout=5),
        base_url="https://x", minio={"endpoint": "h"}, upgrade=True,
    )
    assert removed == []


# --- the manifest ---------------------------------------------------------------------
def _manifest(tmp_path, ids):
    m = Manifest(tmp_path / "manifest.db")
    m.add_papers(PaperRow(arxiv_id=i, version="v1", shard="2301", title=i) for i in ids)
    return m


def _apply(m, *results):
    writer = ManifestWriter(m.db_path)
    writer.start()
    for r in results:
        writer.submit(r)
    writer.stop()


def _done(arxiv_id, converter, md_bytes=10):
    return TaskResult(arxiv_id=arxiv_id, status=DONE, md_bytes=md_bytes, n_pages=1,
                      n_tables=0, n_chars=10, remote_only=True, count_attempt=True,
                      converter=converter)


def test_the_converter_is_recorded_and_only_fallbacks_are_candidates(tmp_path):
    m = _manifest(tmp_path, ["a", "b", "c"])
    _apply(m, _done("a", "pymupdf"), _done("b", "docling"))
    m.mark_done_from_objects([("c", 10, None)])     # another device's paper: converter NULL

    assert m.upgrade_candidates("pymupdf") == ["a"]


def test_a_failed_upgrade_leaves_the_row_untouched_but_counts(tmp_path):
    m = _manifest(tmp_path, ["a"])
    _apply(m, _done("a", "pymupdf", md_bytes=77))
    before = dict(m.conn.execute("SELECT * FROM papers WHERE arxiv_id = 'a'").fetchone())

    _apply(m, TaskResult(arxiv_id="a", status=DONE, upgrade_failed=True))

    after = dict(m.conn.execute("SELECT * FROM papers WHERE arxiv_id = 'a'").fetchone())
    assert after.pop("upgrade_attempts") == before.pop("upgrade_attempts") + 1
    assert after == before                      # status, sizes, flags, converter, timestamp


def test_the_attempt_ceiling_retires_a_paper_the_primary_cannot_convert(tmp_path):
    m = _manifest(tmp_path, ["a"])
    _apply(m, _done("a", "pymupdf"),
           *[TaskResult(arxiv_id="a", status=DONE, upgrade_failed=True)] * 3)

    assert m.upgrade_candidates("pymupdf", max_attempts=3) == []
    assert m.upgrade_candidates("pymupdf", max_attempts=None) == ["a"]


def test_an_older_manifest_gains_the_columns(tmp_path):
    path = tmp_path / "manifest.db"
    # The schema as it stood before these two columns existed.
    old = "\n".join(
        line for line in SCHEMA.splitlines()
        if not line.lstrip().startswith(("converter ", "upgrade_attempts "))
    ).replace("DEFAULT -1, --", "DEFAULT -1  --")
    conn = sqlite3.connect(path)
    conn.executescript(old)
    assert "converter" not in {r[1] for r in conn.execute("PRAGMA table_info(papers)")}
    conn.execute("INSERT INTO papers (arxiv_id, status) VALUES ('a', 'done')")
    conn.commit()
    conn.close()

    with Manifest(path) as m:
        row = m.conn.execute("SELECT converter, upgrade_attempts FROM papers").fetchone()
    assert tuple(row) == (None, 0)


# --- the run --------------------------------------------------------------------------
@pytest.fixture()
def pipeline(tmp_path, monkeypatch):
    cfg = Config()
    cfg.paths = Paths(data_dir=tmp_path)
    cfg.convert.workers = 1
    cfg.crawl.workers = 1
    cfg.crawl.cooldown_seconds = 0
    cfg.paths.ensure()
    monkeypatch.setattr(
        C, "download_one",
        lambda row, session, limiter, data_dir, stop: C.DownloadOutcome(
            path=tmp_path / "tmp" / f"{row.arxiv_id}.pdf", size=10, sha256="x"))
    monkeypatch.setattr(C, "ProcessPoolExecutor", lambda n, **kw: ThreadPoolExecutor(n))
    return cfg


def _stub_conversion(monkeypatch, calls, *, fail=()):
    def convert(row, pdf, data_dir, convert_cfg, upgrade=False, **kw):
        calls.append((row.arxiv_id, upgrade))
        if row.arxiv_id in fail:
            return TaskResult(arxiv_id=row.arxiv_id, status=FAILED_CONVERT,
                              error="docling: still broken", count_attempt=True, worker_id=1)
        return TaskResult(arxiv_id=row.arxiv_id, status=DONE, md_bytes=99, n_pages=1,
                          n_tables=0, n_chars=10, count_attempt=True, worker_id=1,
                          converter=convert_cfg.converter)
    monkeypatch.setattr(C, "convert_and_write", convert)


def _row_of(cfg, arxiv_id):
    with Manifest(cfg.paths.manifest_db) as m:
        return dict(m.conn.execute(
            "SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone())


def test_a_run_re_converts_this_devices_fallback_papers(pipeline, monkeypatch):
    with _manifest(pipeline.paths.data_dir, []) as _:
        pass
    with Manifest(pipeline.paths.manifest_db) as m:
        m.add_papers(PaperRow(arxiv_id=i, version="v1", shard="2301", title=i)
                     for i in ("mine", "theirs", "good", "fresh"))
        _apply(m, _done("mine", "pymupdf"), _done("good", "docling"))
        m.mark_done_from_objects([("theirs", 10, None)])
    calls: list = []
    _stub_conversion(monkeypatch, calls)

    tallies = C.run_pipeline(pipeline)

    # Only the fallback paper this device produced is redone -- and as an upgrade.
    assert sorted(calls) == [("fresh", False), ("mine", True)]
    assert tallies["upgraded"] == 1
    assert tallies["done"] == 1 and tallies["processed"] == 1     # `fresh` alone
    row = _row_of(pipeline, "mine")
    assert (row["status"], row["converter"], row["md_bytes"]) == (DONE, "docling", 99)
    # Nothing left to upgrade, so a third run would not touch it again.
    with Manifest(pipeline.paths.manifest_db) as m:
        assert m.upgrade_candidates("pymupdf") == []


def test_a_paper_the_primary_fails_on_again_stays_done_as_it_was(pipeline, monkeypatch):
    with Manifest(pipeline.paths.manifest_db) as m:
        m.add_papers([PaperRow(arxiv_id="mine", version="v1", shard="2301", title="t")])
        _apply(m, _done("mine", "pymupdf", md_bytes=77))
    calls: list = []
    _stub_conversion(monkeypatch, calls, fail={"mine"})

    tallies = C.run_pipeline(pipeline)

    assert calls == [("mine", True)]            # no in-run retry of an upgrade
    assert tallies["upgrade_failed"] == 1 and tallies["failed"] == 0
    row = _row_of(pipeline, "mine")
    assert (row["status"], row["converter"], row["md_bytes"]) == (DONE, "pymupdf", 77)
    assert row["upgrade_attempts"] == 1 and row["remote_only"] == 1


def test_upgrades_can_be_switched_off(pipeline, monkeypatch):
    pipeline.convert.upgrade_fallbacks = False
    with Manifest(pipeline.paths.manifest_db) as m:
        m.add_papers([PaperRow(arxiv_id="mine", version="v1", shard="2301", title="t")])
        _apply(m, _done("mine", "pymupdf"))
    calls: list = []
    _stub_conversion(monkeypatch, calls)

    C.run_pipeline(pipeline)
    assert calls == []


# --- learning the converter after the fact --------------------------------------------
LOG = """\
2026-09-30 10:00:00 WARNING src.utils.converter: math/0303283 converted by fallback pymupdf after: docling: S3Error
2026-09-30 10:00:01 INFO src.utils.crawler: resuming 2301.00002 from byte 1048576
2026-09-30 10:00:02 WARNING src.utils.converter: 1602.02240 converted by fallback pymupdf after: docling: timeout
"""


def test_fallbacks_are_read_out_of_the_log():
    assert fallbacks_in_log(LOG.splitlines()) == {
        "math/0303283": "pymupdf", "1602.02240": "pymupdf"}


def test_the_log_is_scanned_once_and_only_fills_unknown_done_rows(tmp_path):
    m = _manifest(tmp_path, ["math/0303283", "1602.02240", "2301.00002"])
    _apply(m, TaskResult(arxiv_id="math/0303283", status=DONE, md_bytes=1, n_chars=1),
           _done("1602.02240", "docling"))      # since re-converted: must not be undone
    log_path, state = tmp_path / "crawler.log", tmp_path / "scan.json"
    log_path.write_text(LOG)

    assert scan_log(m, log_path, state) == 1
    assert m.upgrade_candidates("pymupdf") == ["math/0303283"]
    assert scan_log(m, log_path, state) == 0    # nothing new in the log


def test_the_converter_is_read_from_front_matter():
    head = b"---\nid: '2301.00001'\ntitle: T\nconverter: pymupdf\n---\n\nconverter: docling\n"
    assert converter_in_front_matter(head) == "pymupdf"
    assert converter_in_front_matter(b"no front matter\nconverter: x\n") is None


def test_a_bucket_scan_fills_in_this_devices_unknown_papers(tmp_path):
    m = _manifest(tmp_path, ["2301.00001", "2301.00002", "2301.00003"])
    _apply(m, TaskResult(arxiv_id="2301.00001", status=DONE, md_bytes=1, n_chars=1),
           TaskResult(arxiv_id="2301.00002", status=DONE, md_bytes=1, n_chars=1))
    m.mark_done_from_objects([("2301.00003", 10, None)])     # not ours: never examined
    client = FakeClient()
    client.objects = {
        "arxiv/md/2301/2301.00001.md": b"---\nid: x\nconverter: pymupdf\n---\nbody",
        "arxiv/md/2301/2301.00002.md": b"---\nid: y\nconverter: docling\n---\nbody",
        "arxiv/md/2301/2301.00003.md": b"---\nid: z\nconverter: pymupdf\n---\nbody",
    }

    counts = scan_bucket(m, _store(client))

    assert counts == {"examined": 2, "unreadable": 0, "pymupdf": 1, "docling": 1}
    assert m.upgrade_candidates("pymupdf") == ["2301.00001"]
