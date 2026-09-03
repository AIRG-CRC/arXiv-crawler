import pytest

from src.config import Convert
from src.utils.converter import (
    MARKER, ConversionResult, PyMuPDFConverter, TableBlock, accept_fallback,
    clean_cell, get_converter, is_degenerate, lift_tables, looks_like_display_equation,
    looks_like_pseudocode, mark_equations, render_rows, strip_placeholder_columns,
)

pymupdf = pytest.importorskip("pymupdf")


# --- lift_tables: shared by every backend, so it carries the most weight -------------
SAMPLE = """# Results

Prose before.

|Method|Acc|F1|
|---|---|---|
|Ours|91.2|0.88|
|Base|85.0|0.81|

Prose between.

|A|B|
|---|---|
|1|2|

Prose after."""


def test_lift_tables_extracts_and_replaces():
    body, tables = lift_tables(SAMPLE, page=4, start_index=1)
    assert len(tables) == 2
    assert MARKER.format(n=1) in body and MARKER.format(n=2) in body
    assert "|Ours|" not in body            # the table really left the body
    assert "Prose between." in body        # surrounding text is untouched


def test_lift_tables_records_shape_and_page():
    _, tables = lift_tables(SAMPLE, page=4, start_index=1)
    first = tables[0]
    assert (first.index, first.page) == (1, 4)
    assert first.n_cols == 3
    assert first.n_rows == 2               # header excluded
    # Cells are normalised on the way out, for the benefit of retrieval.
    assert first.markdown.splitlines()[0] == "| Method | Acc | F1 |"
    assert first.columns == ["Method", "Acc", "F1"]


def test_lift_tables_continues_numbering_across_pages():
    _, page1 = lift_tables(SAMPLE, page=1, start_index=1)
    _, page2 = lift_tables(SAMPLE, page=2, start_index=len(page1) + 1)
    assert [t.index for t in page1 + page2] == [1, 2, 3, 4]


def test_lift_tables_ignores_pipes_that_are_not_tables():
    """A separator row is required, so prose containing pipes is left alone."""
    text = "Cost is |x| for all x.\nAnother | pipe | line."
    body, tables = lift_tables(text, page=1, start_index=1)
    assert tables == [] and body == text


def test_lift_tables_on_table_only_input():
    body, tables = lift_tables("|A|B|\n|---|---|\n|1|2|", page=1, start_index=1)
    assert len(tables) == 1 and body.strip() == MARKER.format(n=1)


def test_lift_tables_on_empty_input():
    assert lift_tables("", page=1, start_index=1) == ("", [])


# --- fixture PDF ---------------------------------------------------------------------
PROSE = (
    "Transformer architectures have become the dominant approach for a wide range of "
    "sequence modelling problems across several research communities in recent years. "
)


def _write_pdf(path, *, ruled=True):
    """Build a PDF shaped like a real arXiv page: prose around one 3-column table.

    `ruled=True` draws a full grid (what `lines_strict` looks for); `ruled=False` draws
    only horizontal rules, the LaTeX booktabs shape that triggers the fallback.

    The prose matters. A page that is *mostly* table is not representative, and the
    fallback guard is specifically about not letting a retry swallow a real body.
    """
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 80), "Introduction", fontsize=16)
    page.insert_textbox(
        pymupdf.Rect(72, 95, 520, 300), PROSE * 6, fontsize=10, lineheight=1.2
    )

    xs, ys = [72, 190, 300, 420], [330, 360, 390, 420]
    rows = [["Method", "Acc", "F1"], ["Ours", "91.2", "0.88"], ["Base", "85.0", "0.81"]]
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            page.insert_text((xs[c] + 6, ys[r] + 20), cell, fontsize=11)
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y), width=0.8)
    if ruled:
        for x in xs:
            page.draw_line((x, ys[0]), (x, ys[-1]), width=0.8)

    page.insert_textbox(
        pymupdf.Rect(72, 470, 520, 700), PROSE * 6, fontsize=10, lineheight=1.2
    )
    doc.save(path)
    doc.close()
    return path


@pytest.fixture()
def ruled_pdf(tmp_path):
    return _write_pdf(tmp_path / "ruled.pdf", ruled=True)


@pytest.fixture()
def booktabs_pdf(tmp_path):
    return _write_pdf(tmp_path / "booktabs.pdf", ruled=False)


def _cfg(**kw):
    return Convert(**{"max_pages": 50, "min_chars_per_page": 100, **kw})


# --- PyMuPDF backend -----------------------------------------------------------------
pymupdf4llm = pytest.importorskip("pymupdf4llm", reason="default backend not installed")


def test_pymupdf_backend_extracts_body_and_table(ruled_pdf):
    result = get_converter("pymupdf", _cfg()).convert(ruled_pdf)
    assert isinstance(result, ConversionResult)
    assert result.n_pages == 1
    assert "Introduction" in result.body_markdown
    assert result.n_tables == 1

    table = result.tables[0]
    assert table.n_cols == 3
    assert "Method" in table.markdown and "91.2" in table.markdown
    # The table's content belongs to the tables file, not the body.
    assert "91.2" not in result.body_markdown
    assert MARKER.format(n=1) in result.body_markdown


def test_rule_only_tables_are_the_known_gap(booktabs_pdf):
    """A booktabs table -- horizontal rules, no verticals -- is not found by
    lines_strict. This is a documented limitation, asserted so it is visible."""
    assert get_converter("pymupdf", _cfg()).convert(booktabs_pdf).n_tables == 0


def test_text_fallback_is_rejected_rather_than_eating_the_body(booktabs_pdf):
    """The "text" strategy collapses an entire page into one table: on this fixture it
    takes the body from ~2,000 characters to 11. The guard must keep the first pass,
    even though that leaves the rule-only table unrecovered."""
    result = get_converter("pymupdf", _cfg(table_fallback_strategy="text")).convert(booktabs_pdf)
    assert result.n_tables == 0
    assert result.n_chars > 1000
    assert "Transformer architectures" in result.body_markdown


def test_max_pages_truncates_rather_than_failing(tmp_path):
    doc = pymupdf.open()
    for i in range(5):
        doc.new_page().insert_text((72, 80), f"Page {i} body text.", fontsize=12)
    path = tmp_path / "long.pdf"
    doc.save(path)
    doc.close()

    result = get_converter("pymupdf", _cfg(max_pages=2)).convert(path)
    assert result.n_pages == 5 and result.truncated is True
    assert "Page 4" not in result.body_markdown


def test_fallback_is_rejected_when_it_swallows_the_body():
    """The real failure mode: on a 75-page paper the "text" strategy reported 70
    tables while collapsing the body from 252,000 characters to 3,000."""
    first = ConversionResult(body_markdown="x", n_chars=251_969)
    retry = ConversionResult(
        body_markdown="y", n_chars=3_047,
        tables=[TableBlock(index=i, page=i, n_rows=2, n_cols=2, markdown="|a|\n|---|")
                for i in range(1, 71)],
    )
    assert accept_fallback(first, retry) is False


def test_fallback_is_accepted_when_the_body_survives():
    first = ConversionResult(body_markdown="x", n_chars=10_000)
    retry = ConversionResult(
        body_markdown="y", n_chars=9_200,
        tables=[TableBlock(index=1, page=1, n_rows=2, n_cols=3, markdown="|a|\n|---|")],
    )
    assert accept_fallback(first, retry) is True


def test_fallback_with_no_tables_is_never_taken():
    first = ConversionResult(body_markdown="x", n_chars=10_000)
    assert accept_fallback(first, ConversionResult(body_markdown="y", n_chars=10_000)) is False


def test_unknown_converter_names_the_valid_options():
    with pytest.raises(ValueError, match="unknown converter"):
        get_converter("nope", _cfg())


# --- cell normalisation: the RAG-facing half of table extraction ---------------------
def test_clean_cell_flattens_wrapped_lines():
    """PDF extraction packs wrapped text into one cell joined by <br>; left alone that
    is markup soup for an embedding model to wade through."""
    assert clean_cell("Vinyals &amp;<br>Kaiser<br/>el al.") == "Vinyals &amp; Kaiser el al."
    assert clean_cell("  spaced   out  ") == "spaced out"
    assert clean_cell("a|b") == "a\\|b"          # a stray pipe cannot break the row
    assert clean_cell("") == ""


def test_render_rows_pads_ragged_rows_rectangular():
    md = render_rows([["A", "B", "C"], ["1"], ["2", "3"]])
    assert [ln.count("|") for ln in md.splitlines()] == [4, 4, 4, 4]


# --- degenerate-table filter ---------------------------------------------------------
def test_header_only_table_is_degenerate():
    assert is_degenerate([["Method", "Acc"]]) is True


def test_all_empty_table_is_degenerate():
    assert is_degenerate([["", ""], ["", ""]]) is True


def test_absurdly_wide_table_is_degenerate():
    """A 4x59 attention-map figure came back as a 'table' on a real paper."""
    wide = [[f"c{i}" for i in range(59)] for _ in range(4)]
    assert is_degenerate(wide, max_columns=25) is True
    assert is_degenerate(wide, max_columns=80) is False


def test_placeholder_header_table_is_degenerate():
    """pymupdf4llm emits Col1..ColN only when it found no header row. Measured share:
    genuine tables 0-33%, misparsed figure legends 80-90%."""
    junk = [["MNIST Logistic Regression"] + [f"Col{i}" for i in range(2, 11)],
            ["", "", "", "", "", "", "", "Ad SG", "aGrad", "erov"]]
    assert is_degenerate(junk) is True


def test_real_table_survives_every_filter():
    good = [["Method", "Acc", "F1"], ["Ours", "91.2", "0.88"], ["Base", "85.0", "0.81"]]
    assert is_degenerate(good) is False


def test_degenerate_tables_are_dropped_and_numbering_stays_contiguous():
    md = ("|Method|Acc|\n|---|---|\n|Ours|91.2|\n\n"
          "text\n\n"
          "|Head|Only|\n|---|---|\n\n"          # header-only -> dropped
          "more\n\n"
          "|A|B|\n|---|---|\n|1|2|")
    body, tables = lift_tables(md, page=1, start_index=1)
    assert [t.index for t in tables] == [1, 2]
    assert MARKER.format(n=3) not in body        # no marker for the dropped table
    assert body.count("[[TABLE:") == 2


def test_placeholder_columns_are_stripped_from_metadata():
    assert strip_placeholder_columns(["Method", "Col2", "Col3", "F1"]) == ["Method", "F1"]


# --- pseudocode ----------------------------------------------------------------------
ALGO_ROWS = [
    ["Algorithm 1 Adam"],
    ["1: Require: alpha: Stepsize"],
    ["2: while theta not converged do"],
    ["3: t = t + 1"],
    ["4: end while"],
    ["5: return theta"],
]


def test_algorithm_float_is_recognised_not_treated_as_a_table():
    """algorithm2e draws a ruled box, which is exactly the geometry table detection
    keys on -- so algorithm floats routinely arrive as tables."""
    assert looks_like_pseudocode(ALGO_ROWS, caption="") is True


def test_algorithm_caption_alone_is_enough():
    assert looks_like_pseudocode([["x", "y"], ["1", "2"], ["3", "4"]],
                                 caption="Algorithm 2: Adam optimiser") is True


def test_ordinary_table_is_not_mistaken_for_pseudocode():
    rows = [["Method", "Acc", "F1"], ["Ours", "91.2", "0.88"], ["Base", "85.0", "0.81"]]
    assert looks_like_pseudocode(rows, caption="Table 3: results") is False


def test_pseudocode_is_emitted_as_fenced_code_preserving_line_order():
    md = "|" + "|\n|".join("Algorithm 1 Adam ---  1: Require: alpha  "
                           "2: while theta not converged do  3: t = t + 1  "
                           "4: end while  5: return theta".split("  ")) + "|"
    lines = ["|Algorithm 1 Adam|", "|---|", "|1: Require: alpha|",
             "|2: while theta not converged do|", "|3: t = t+1|",
             "|4: end while|", "|5: return theta|"]
    body, tables = lift_tables("\n".join(lines), page=3, start_index=1)
    assert len(tables) == 1
    block = tables[0]
    assert block.kind == "pseudocode"
    assert block.markdown.startswith("```text")
    assert "2: while theta not converged do" in block.markdown
    assert "|" not in block.markdown.replace("```text", "")   # no pipe-table mangling


def test_pseudocode_detection_can_be_switched_off():
    lines = ["|Algorithm 1 Adam|", "|---|", "|1: Require: alpha|",
             "|2: while theta not converged do|", "|3: t = t+1|", "|4: end while|"]
    _, tables = lift_tables("\n".join(lines), page=3, start_index=1,
                            detect_pseudocode=False)
    assert tables and tables[0].kind == "table"


# --- captions ------------------------------------------------------------------------
def test_caption_below_the_table_is_captured():
    md = "|A|B|\n|---|---|\n|1|2|\n\nTable 4: The Transformer generalizes well."
    _, tables = lift_tables(md, page=2, start_index=1)
    assert tables[0].caption == "Table 4: The Transformer generalizes well."


def test_caption_above_the_table_is_captured():
    md = "Table 1: Ablation results.\n\n|A|B|\n|---|---|\n|1|2|"
    _, tables = lift_tables(md, page=2, start_index=1)
    assert tables[0].caption == "Table 1: Ablation results."


def test_unrelated_prose_is_not_taken_as_a_caption():
    md = "We now describe the method.\n\n|A|B|\n|---|---|\n|1|2|"
    _, tables = lift_tables(md, page=2, start_index=1)
    assert tables[0].caption == ""


# --- equations -----------------------------------------------------------------------
def test_display_equations_are_marked():
    assert looks_like_display_equation("Attention(Q, K, V ) = softmax(QK^T/√dk)V") is True
    assert looks_like_display_equation("α = β + γ  (3)") is True


def test_prose_is_not_marked_as_an_equation():
    assert looks_like_display_equation(
        "The encoder is composed of a stack of six identical layers.") is False
    assert looks_like_display_equation("") is False
    assert looks_like_display_equation("## Results") is False


def test_mark_equations_wraps_and_leaves_prose_alone():
    text = "We define the score.\n\nα = β + γ  (3)\n\nThe result follows."
    out = mark_equations(text)
    assert "$$ α = β + γ  (3) $$" in out
    assert "We define the score." in out
    assert "$$ We define" not in out


def test_mark_equations_leaves_code_fences_alone():
    text = "```text\n1: x = y + z\n```"
    assert mark_equations(text) == text


# --- equation false positives, each seen in real converted output --------------------
@pytest.mark.parametrize("line", [
    "_⊙_", "_←_", "_∇_", "_∈_",                                   # stranded glyphs
    "of continuous representations **z** = ( _z_ 1 _, ..., zn_ ). Given **z**, "
    "the decoder then generates an output",                       # prose with maths in it
    "square _gt_ _gt_ . Good default settings for the tested machine learning "
    "problems are _α_ = 0 _._ 001,",
    "In this work we employ _h_ = 8 parallel attention layers, or heads.",
])
def test_equation_false_positives_are_rejected(line):
    """Markdown italics inflated the non-letter ratio, so whole prose sentences were
    being wrapped in $$; lone glyphs were too. Both were seen in real output."""
    assert looks_like_display_equation(line) is False


@pytest.mark.parametrize("line", [
    "Attention( _Q, K, V_ ) = softmax( _[QK]_ ~~_√_~~ _[T]_",
    "MultiHead( _Q, K, V_ ) = Concat(head1 _, ...,_ headh) _W_ _[O]_",
    "FFN( _x_ ) = max(0 _, xW_ 1 + _b_ 1) _W_ 2 + _b_ 2 (2)",
    "_t ←_ _t_ + 1",
])
def test_real_equations_survive_the_stricter_rules(line):
    assert looks_like_display_equation(line) is True
