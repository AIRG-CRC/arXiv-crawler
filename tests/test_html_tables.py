"""HTML tables from LightOnOCR-2, rendered as pipe tables.

LightOnOCR-2 emits tables as HTML on purpose -- "some nested tables cannot be represented
in markdown" -- but every other backend here produces pipe tables, and `lift_tables` only
recognises those. Converting on the way in keeps one backend's papers from being shaped
differently from the rest of the corpus.
"""

from src.utils.converter import html_table_to_rows, html_tables_to_pipe, lift_tables

SIMPLE = """<table>
<tr><th>Method</th><th>Acc</th></tr>
<tr><td>Ours</td><td>91.2</td></tr>
<tr><td>Base</td><td>85.0</td></tr>
</table>"""


def test_a_plain_table_becomes_rows():
    assert html_table_to_rows(SIMPLE) == [
        ["Method", "Acc"], ["Ours", "91.2"], ["Base", "85.0"],
    ]


def test_conversion_produces_a_pipe_table_lift_tables_can_see():
    body, tables = lift_tables(html_tables_to_pipe(SIMPLE), page=1, start_index=1)
    assert len(tables) == 1
    assert tables[0].columns == ["Method", "Acc"]
    assert tables[0].n_rows == 2 and tables[0].n_cols == 2
    assert "<table>" not in body


def test_colspan_is_repeated_so_rows_stay_rectangular():
    html = ('<table><tr><th colspan="2">Results</th></tr>'
            '<tr><td>a</td><td>b</td></tr></table>')
    assert html_table_to_rows(html) == [["Results", "Results"], ["a", "b"]]


def test_entities_and_inline_markup_are_flattened():
    html = ("<table><tr><td><b>F&amp;M</b></td><td>a<br/>b</td></tr>"
            "<tr><td>x</td><td>y</td></tr></table>")
    assert html_table_to_rows(html) == [["F&M", "a b"], ["x", "y"]]


def test_latex_spans_inside_cells_survive():
    """The reason to use this backend at all is that the maths comes back as maths."""
    html = (r'<table><tr><th>Loss</th></tr><tr><td>$\mathcal{L} = -\log p$</td></tr></table>')
    rows = html_table_to_rows(html)
    assert rows[1] == [r"$\mathcal{L} = -\log p$"]


def test_several_tables_in_one_page_are_all_converted():
    page = f"Intro.\n\n{SIMPLE}\n\nMiddle.\n\n{SIMPLE}\n\nEnd."
    converted = html_tables_to_pipe(page)
    assert "<table>" not in converted
    _, tables = lift_tables(converted, page=1, start_index=1)
    assert len(tables) == 2
    assert "Intro." in converted and "End." in converted


def test_a_table_that_flattens_to_nothing_is_left_alone():
    """Better to pass the original through than to silently emit an empty table."""
    html = "<table><tr><td></td></tr></table>"
    assert html_tables_to_pipe(html) == html


def test_prose_without_tables_is_untouched():
    text = "No tables here, just 1 < 2 and a > b."
    assert html_tables_to_pipe(text) == text
