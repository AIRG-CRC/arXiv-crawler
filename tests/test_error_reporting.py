"""How a conversion failure is described.

Deliberately free of `importorskip`: `test_converter.py` gates its whole module on
`pymupdf4llm`, so on a machine running the docling backend every test in it is skipped.
These assertions are pure logic and must run everywhere -- they are the reason a failed
paper is diagnosable at all.
"""

import tempfile
from pathlib import Path

import pytest

from src.config import Convert
from src.utils.converter import (
    REGISTRY, BaseConverter, ConversionResult, _converter_chain, convert_and_write,
    describe_exception,
)
from src.utils.state import DONE, FAILED_CONVERT, PaperRow


def test_describe_exception_unwraps_a_pipeline_wrapper():
    """docling raises `RuntimeError("Pipeline X failed") from e` and logs nothing.

    Recording only the outer message makes every such failure indistinguishable, which
    is exactly what left three papers unexplainable across two runs.
    """
    try:
        try:
            raise ValueError("Output queue closed while emitting from layout")
        except ValueError as inner:
            raise RuntimeError("Pipeline StandardPdfPipeline failed") from inner
    except RuntimeError as exc:
        described = describe_exception(exc)

    assert described.startswith("RuntimeError: Pipeline StandardPdfPipeline failed")
    assert "ValueError: Output queue closed while emitting from layout" in described


def test_describe_exception_respects_a_suppressed_context():
    """`raise ... from None` means "the earlier error is noise" -- honour that."""
    try:
        try:
            raise KeyError("incidental")
        except KeyError:
            raise TypeError("the real one") from None
    except TypeError as exc:
        assert describe_exception(exc) == "TypeError: the real one"


def test_describe_exception_follows_an_implicit_context():
    """An error raised *during* handling of another still explains itself."""
    try:
        try:
            raise ValueError("the underlying problem")
        except ValueError:
            raise RuntimeError("the wrapper")
    except RuntimeError as exc:
        assert "ValueError: the underlying problem" in describe_exception(exc)


def test_describe_exception_survives_a_cycle():
    a = ValueError("a")
    b = ValueError("b")
    a.__cause__ = b
    b.__cause__ = a                      # a chain that would otherwise loop forever
    assert describe_exception(a) == "ValueError: a <- caused by ValueError: b"


def test_describe_exception_is_bounded():
    exc = deepest = ValueError("level 0")
    for i in range(1, 10):
        wrapper = ValueError(f"level {i}")
        wrapper.__cause__ = deepest
        deepest = wrapper
    assert describe_exception(deepest, limit=2).count("caused by") == 1
    assert exc is not None


# --- fallback chain ------------------------------------------------------------------
def test_chain_is_just_the_primary_when_no_fallback_is_configured():
    chain = _converter_chain(Convert(converter="docling", fallback_converter=None))
    assert chain == [("docling", True)]


def test_chain_adds_the_fallback_and_marks_the_last_chance():
    chain = _converter_chain(Convert(converter="docling", fallback_converter="pymupdf"))
    assert chain == [("docling", False), ("pymupdf", True)]


def test_a_fallback_equal_to_the_primary_is_not_tried_twice():
    chain = _converter_chain(Convert(converter="pymupdf", fallback_converter="pymupdf"))
    assert chain == [("pymupdf", True)]


class _Boom(BaseConverter):
    name = "_boom"

    def convert(self, pdf_path):
        raise RuntimeError("primary exploded")


class _Fine(BaseConverter):
    name = "_fine"

    def convert(self, pdf_path):
        return ConversionResult(body_markdown="Body.", n_pages=2, n_chars=5)


@pytest.fixture()
def stub_backends():
    REGISTRY["_boom"], REGISTRY["_fine"] = _Boom, _Fine
    yield
    del REGISTRY["_boom"], REGISTRY["_fine"]


def _row():
    return PaperRow(arxiv_id="2301.00001", version="v1", shard="2301", title="T",
                    authors='["A, B"]', categories="cs.LG", primary_category="cs.LG")


def test_fallback_rescues_a_paper_the_primary_cannot_convert(stub_backends):
    tmp = Path(tempfile.mkdtemp())
    pdf = tmp / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    result = convert_and_write(
        _row(), pdf, tmp,
        Convert(converter="_boom", fallback_converter="_fine", timeout=5),
        base_url="https://x",
    )
    assert result.status == DONE
    # The backend that actually ran is recorded, not the one that was configured.
    assert result.converter == "_fine"
    assert "converter: _fine" in (tmp / "md" / "2301" / "2301.00001.md").read_text()
    assert not pdf.exists()            # still cleaned up exactly once, after both tries


def test_both_failing_reports_both_errors_and_keeps_the_paper_failed(stub_backends):
    tmp = Path(tempfile.mkdtemp())
    pdf = tmp / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    result = convert_and_write(
        _row(), pdf, tmp,
        Convert(converter="_boom", fallback_converter="_boom2", timeout=5),
        base_url="https://x",
    )
    assert result.status == FAILED_CONVERT
    assert "_boom" in result.error and "primary exploded" in result.error
    assert "unknown converter" in result.error      # the second name does not exist


# --- a converter that fails on everything ---------------------------------------------
def test_a_systematic_fallback_is_announced_once_then_goes_quiet(tmp_path, monkeypatch, capsys):
    """One line per paper is the wrong shape for a broken install.

    With docling unable to fetch its model, every paper takes the fallback -- so the old
    per-paper line printed once per paper for the whole corpus, and the run summary was
    indistinguishable from a healthy one.
    """
    from concurrent.futures import ThreadPoolExecutor

    from src.config import Config, Paths
    from src.utils import crawler as C
    from src.utils.state import DONE, Manifest, PaperRow, TaskResult

    said: list[str] = []
    monkeypatch.setattr(C, "console", lambda msg, *a: said.append(msg % a if a else msg))

    cfg = Config()
    cfg.paths = Paths(data_dir=tmp_path)
    cfg.convert.workers = 1
    cfg.crawl.workers = 1
    cfg.crawl.cooldown_seconds = 0
    cfg.convert.converter = "docling"
    cfg.paths.ensure()

    ids = [f"2301.{i:05d}" for i in range(30)]
    with Manifest(cfg.paths.manifest_db) as m:
        m.add_papers([PaperRow(arxiv_id=i, version="v1", shard="2301") for i in ids])

    monkeypatch.setattr(C, "download_one", lambda row, s, l, d, stop: C.DownloadOutcome(
        path=tmp_path / "tmp" / f"{row.arxiv_id}.pdf", size=10, sha256="x"))
    monkeypatch.setattr(C, "ProcessPoolExecutor", lambda n, **kw: ThreadPoolExecutor(n))
    monkeypatch.setattr(C, "convert_and_write", lambda row, pdf, d, c, **kw: TaskResult(
        arxiv_id=row.arxiv_id, status=DONE, md_bytes=10, n_pages=1, n_tables=0, n_chars=10,
        count_attempt=True, worker_id=1, converter="pymupdf"))

    tallies = C.run_pipeline(cfg)

    assert tallies["fell_back"] == len(ids)
    per_paper = [line for line in said if "converted by fallback" in line]
    alarms = [line for line in said if "consecutive papers" in line]
    assert len(alarms) == 1, "the alarm should be raised exactly once"
    assert len(per_paper) == C.FALLBACK_ALARM - 1, "the per-paper line must stop at the alarm"
    assert "docling" in alarms[0] and "pymupdf" in alarms[0]


def test_an_occasional_fallback_does_not_raise_the_alarm(tmp_path, monkeypatch):
    """A run of awkward PDFs is normal; the counter has to reset on every good paper."""
    from concurrent.futures import ThreadPoolExecutor

    from src.config import Config, Paths
    from src.utils import crawler as C
    from src.utils.state import DONE, Manifest, PaperRow, TaskResult

    said: list[str] = []
    monkeypatch.setattr(C, "console", lambda msg, *a: said.append(msg % a if a else msg))

    cfg = Config()
    cfg.paths = Paths(data_dir=tmp_path)
    cfg.convert.workers = 1
    cfg.crawl.workers = 1
    cfg.crawl.cooldown_seconds = 0
    cfg.convert.converter = "docling"
    cfg.paths.ensure()

    ids = [f"2301.{i:05d}" for i in range(30)]
    with Manifest(cfg.paths.manifest_db) as m:
        m.add_papers([PaperRow(arxiv_id=i, version="v1", shard="2301") for i in ids])

    seen = {"n": 0}

    def convert(row, pdf, d, c, **kw):
        seen["n"] += 1
        # every third paper needs the fallback; the rest are fine
        backend = "pymupdf" if seen["n"] % 3 == 0 else "docling"
        return TaskResult(arxiv_id=row.arxiv_id, status=DONE, md_bytes=10, n_pages=1,
                          n_tables=0, n_chars=10, count_attempt=True, worker_id=1,
                          converter=backend)

    monkeypatch.setattr(C, "download_one", lambda row, s, l, d, stop: C.DownloadOutcome(
        path=tmp_path / "tmp" / f"{row.arxiv_id}.pdf", size=10, sha256="x"))
    monkeypatch.setattr(C, "ProcessPoolExecutor", lambda n, **kw: ThreadPoolExecutor(n))
    monkeypatch.setattr(C, "convert_and_write", convert)

    C.run_pipeline(cfg)
    assert not [line for line in said if "consecutive papers" in line]


def test_a_multi_paragraph_error_is_clipped_for_the_terminal(tmp_path, monkeypatch):
    """A HuggingFace download failure runs to hundreds of characters over several lines."""
    from concurrent.futures import ThreadPoolExecutor

    from src.config import Config, Paths
    from src.utils import crawler as C
    from src.utils.state import FAILED_CONVERT, Manifest, PaperRow, TaskResult

    said: list[str] = []
    monkeypatch.setattr(C, "console", lambda msg, *a: said.append(msg % a if a else msg))

    cfg = Config()
    cfg.paths = Paths(data_dir=tmp_path)
    cfg.convert.workers = 1
    cfg.crawl.workers = 1
    cfg.crawl.cooldown_seconds = 0
    cfg.retry.in_run = False
    cfg.paths.ensure()

    with Manifest(cfg.paths.manifest_db) as m:
        m.add_papers([PaperRow(arxiv_id="2301.00001", version="v1", shard="2301")])

    sprawling = ("docling: RepositoryNotFoundError: 401 Client Error.\n\n"
                 "Repository Not Found for url: https://huggingface.co/api/models/x.\n"
                 "Please make sure you specified the correct repo_id and repo_type.\n"
                 "If you are trying to access a private or gated repo, make sure you are "
                 "authenticated and your token has the required permissions.\n"
                 "For more details, see https://huggingface.co/docs/...")

    monkeypatch.setattr(C, "download_one", lambda row, s, l, d, stop: C.DownloadOutcome(
        path=tmp_path / "tmp" / f"{row.arxiv_id}.pdf", size=10, sha256="x"))
    monkeypatch.setattr(C, "ProcessPoolExecutor", lambda n, **kw: ThreadPoolExecutor(n))
    monkeypatch.setattr(C, "convert_and_write", lambda row, pdf, d, c, **kw: TaskResult(
        arxiv_id=row.arxiv_id, status=FAILED_CONVERT, error=sprawling,
        count_attempt=True, worker_id=1))

    C.run_pipeline(cfg)

    failures = [line for line in said if line.startswith("  ✗")]
    assert len(failures) == 1
    assert "\n" not in failures[0], "a newline in the bar's output shreds the bar"
    assert len(failures[0]) < 230
    assert failures[0].endswith("...")
    # the manifest still keeps the detail
    with Manifest(cfg.paths.manifest_db) as m:
        stored = m.conn.execute("SELECT error FROM papers").fetchone()[0]
        assert "Repository Not Found" in stored


# --- multi-backend failures -------------------------------------------------------------
def test_every_backend_tried_survives_the_length_limit():
    """Which backends were tried is the actionable part, and it used to be truncated away.

    A docling CUDA failure repeats the same cuDNN line four times and runs past 500
    characters unaided, so joining then clipping erased the fallback's error completely --
    and with it the answer to "could anything have converted this paper?".
    """
    from src.utils.converter import join_backend_errors

    verbose = "docling: ConversionError: " + ("CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED; " * 20)
    fallback = "pymupdf: ValueError: cannot open broken document"

    assert "pymupdf" not in (" | then ".join([verbose, fallback]))[:500]   # the old way
    joined = join_backend_errors([verbose, fallback])
    assert "docling" in joined and "pymupdf" in joined
    assert len(joined) <= 500


def test_backend_errors_are_flattened_to_one_line():
    from src.utils.converter import join_backend_errors

    joined = join_backend_errors(["docling: boom\n\nRepository Not Found\n  for url: x"])
    assert "\n" not in joined
    assert "docling: boom Repository Not Found for url: x" == joined


def test_a_single_backend_keeps_its_whole_budget():
    from src.utils.converter import join_backend_errors

    one = "pymupdf: " + "x" * 400
    assert join_backend_errors([one]) == one[:500]


def test_no_errors_is_an_empty_string():
    from src.utils.converter import join_backend_errors

    assert join_backend_errors([]) == ""
    assert join_backend_errors(["", "   "]) == ""


def test_three_backends_all_appear():
    from src.utils.converter import join_backend_errors

    joined = join_backend_errors([f"{name}: " + "y" * 300
                                  for name in ("docling", "pymupdf", "pdfplumber")])
    assert all(name in joined for name in ("docling", "pymupdf", "pdfplumber"))
    assert len(joined) <= 500
