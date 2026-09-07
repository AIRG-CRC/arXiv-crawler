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
