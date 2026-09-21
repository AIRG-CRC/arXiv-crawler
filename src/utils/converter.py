"""PDF -> Markdown backends.

Every backend returns the same `ConversionResult`: a body with each table lifted out
and replaced by a ``[[TABLE:n]]`` marker, plus the tables themselves. That split is what
lets the body and the tables land in separate files without losing where a table sat in
the text.

Two things happen to a table on the way out, both for the benefit of retrieval:

  * cells are normalised -- ``<br>`` soup flattened, whitespace collapsed, rows padded
    rectangular -- so a chunk of the table is readable prose-adjacent text, not markup;
  * its caption and column names are captured, so the tables file can carry enough
    context for each table to stand alone as a retrieval chunk.

Backends import their heavy dependency *inside* `__init__`, so a missing `docling` or an
absent JVM only raises if you actually select that backend.
"""

from __future__ import annotations

import inspect
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# A GitHub pipe-table row, and the `|---|---|` separator that follows the header.
_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_SEP_LINE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")

# "Table 3:", "TABLE II -", "Table 1." -- the caption line beside a float.
_CAPTION = re.compile(r"^\s*(?:Table|TABLE|Tab\.)\s*([0-9]+|[IVXLC]+)\s*[.:)\-—]?\s*(.*)$")
_ALGO_CAPTION = re.compile(r"^\s*(?:Algorithm|ALGORITHM|Alg\.|Procedure)\s*([0-9]+|[IVXLC]+)?\b", re.I)

# Lines that look like pseudocode rather than tabular data.
_ALGO_KEYWORD = re.compile(
    r"\b(?:Require|Ensure|Input|Output|for\s+each|end\s+(?:for|while|if|procedure|function)"
    r"|while|repeat|until|procedure|function|return)\b:?", re.I
)
_ALGO_LINE_NUMBER = re.compile(r"^\s*\d+\s*[:.]\s+\S")

MARKER = "[[TABLE:{n}]]"
CAPTION_LOOKAROUND = 4        # lines either side of a table to search for its caption


# --- cell / table normalisation ------------------------------------------------------
def split_row(line: str) -> list[str]:
    """Split one pipe-table line into its cells."""
    cells = line.strip().split("|")
    if cells and not cells[0].strip():
        cells = cells[1:]
    if cells and not cells[-1].strip():
        cells = cells[:-1]
    return cells


def clean_cell(cell: str) -> str:
    """Flatten a cell to a single readable line.

    PDF table extraction packs wrapped lines into one cell joined by ``<br>``. Left
    alone that produces markup soup an embedding model has to wade through, so it is
    flattened to spaces. Pipes are escaped so a stray one cannot break the row.
    """
    text = re.sub(r"<br\s*/?>", " ", cell, flags=re.I)
    text = text.replace("|", "\\|")
    return " ".join(text.split())


def render_rows(rows: list[list[str]]) -> str:
    """Render cleaned rows as a rectangular GitHub pipe table."""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]
    head, *body = padded
    out = ["| " + " | ".join(head) + " |", "|" + "|".join([" --- "] * width) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


# A table wider than this is almost always a misparse -- attention-map figures and
# multi-column page layouts routinely come back as 30-60 "columns".
DEFAULT_MAX_TABLE_COLUMNS = 25

# pymupdf4llm names columns Col1..ColN only when it could not find a real header row,
# which correlates strongly with "this was never a table". Measured across three papers:
# genuine tables ran 0-33% placeholder headers, misparsed figure legends 80-90%.
_PLACEHOLDER_HEADER = re.compile(r"^Col\d+$")
PLACEHOLDER_HEADER_LIMIT = 0.6


def strip_placeholder_columns(columns: list[str]) -> list[str]:
    """Drop auto-generated ColN names -- they tell a retriever nothing."""
    return [c for c in columns if c and not _PLACEHOLDER_HEADER.match(c)]


def _placeholder_share(header: list[str]) -> float:
    if not header:
        return 0.0
    return sum(1 for c in header if _PLACEHOLDER_HEADER.match(c.strip())) / len(header)


def is_degenerate(rows: list[list[str]], *, max_columns: int = DEFAULT_MAX_TABLE_COLUMNS) -> bool:
    """Is this extracted table pure noise?

    Retrieval corpora are damaged more by junk chunks than by missing ones: a
    header-only table, or a figure misread as sixty columns of empty cells, embeds to
    nothing useful and dilutes every real result. Observed on one paper alone -- a
    "0 rows x 5 columns" header fragment and a "4 rows x 59 columns" attention figure.
    """
    if len(rows) < 2:                                   # header with no data under it
        return True
    if not any(cell.strip() for row in rows for cell in row):
        return True
    if max((len(r) for r in rows), default=0) > max_columns:
        return True
    if _placeholder_share(rows[0]) >= PLACEHOLDER_HEADER_LIMIT:
        return True
    # Mostly-empty grids: real tables are not 85% blank.
    cells = [c for row in rows for c in row]
    filled = sum(1 for c in cells if c.strip())
    return bool(cells) and filled / len(cells) < 0.15


def looks_like_pseudocode(rows: list[list[str]], caption: str) -> bool:
    """Is this "table" actually an algorithm float?

    `algorithm2e` and friends draw a boxed float with horizontal rules, which is exactly
    the geometry table detection keys on -- so algorithm blocks routinely come out as
    tables. Rendering an algorithm as a pipe table destroys its indentation and line
    numbering, and gives a retriever a table with no tabular meaning.
    """
    if _ALGO_CAPTION.match(caption or ""):
        return True

    flat = [" ".join(c for c in row if c).strip() for row in rows]
    flat = [line for line in flat if line]
    if len(flat) < 3:
        return False

    numbered = sum(1 for line in flat if _ALGO_LINE_NUMBER.match(line))
    keyworded = sum(1 for line in flat if _ALGO_KEYWORD.search(line))
    # Mostly-one-column content with step numbering or algorithm keywords.
    narrow = max((len(r) for r in rows), default=0) <= 2
    return narrow and (numbered >= len(flat) * 0.5 or keyworded >= 3)


def render_pseudocode(rows: list[list[str]]) -> str:
    """Emit an algorithm block as fenced text, preserving line order."""
    lines = [" ".join(c for c in row if c).rstrip() for row in rows]
    return "```text\n" + "\n".join(line for line in lines if line) + "\n```"


# --- equations -----------------------------------------------------------------------
# A relation or operator is *required*: a Greek letter alone appears constantly in
# ordinary prose ("where α is the learning rate"), so it cannot carry the decision.
_MATH_OPERATORS = set("=≤≥≈≠≡∝±∓∑∏∫√∂∇→←↦⊗⊕⊙∈∉⊂⊆∀∃×·")
# Supporting symbols: they raise confidence but never decide on their own.
_MATH_SYMBOLS = _MATH_OPERATORS | set("αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ∞^_")
_EQUATION_NUMBER = re.compile(r"\s*\(\s*\d+\s*\)\s*$")
MAX_EQUATION_LETTER_RATIO = 0.75
MIN_EQUATION_LENGTH = 6
# Markdown emphasis the converters sprinkle through maths-heavy text. It is markup, and
# counting it as "non-letter" made prose lines look like equations.
_EMPHASIS = re.compile(r"[_*~`\[\]]+")
# Three real words in a row: this is a sentence, not a formula.
_PROSE_RUN = re.compile(r"(?:\b[A-Za-z]{3,}\b[ ,]+){2}\b[A-Za-z]{3,}\b")


def looks_like_display_equation(line: str) -> bool:
    """Conservative test for a standalone display equation.

    Three rules earn their place, each from a false positive seen on real papers:

    * markdown emphasis is stripped before measuring -- the ``_x_`` italics the
      converters emit are markup, not maths, and counting them as non-letters made
      ordinary prose look like an equation;
    * a line must have some substance left after that, or lone stranded glyphs get
      marked (``$$ ⊙ $$``, ``$$ ← $$``);
    * a run of consecutive real words means it is a sentence that happens to contain
      maths, and wrapping prose in ``$$`` mislabels it for anything reading downstream.
    """
    stripped = line.strip()
    if not (3 <= len(stripped) <= 200):
        return False
    if stripped.startswith(("#", ">", "|", "-", "*", "```")):
        return False

    core = _EMPHASIS.sub("", stripped).strip()
    if len(core) < MIN_EQUATION_LENGTH:
        return False
    if not _MATH_OPERATORS & set(core):
        return False
    if _PROSE_RUN.search(core):
        return False
    if core.endswith((".", ":", ";", "?")) and not _EQUATION_NUMBER.search(core):
        return False
    letters = sum(c.isalpha() for c in core)
    return (letters / len(core) < MAX_EQUATION_LETTER_RATIO
            and len(core.split()) <= 25)


def mark_equations(markdown: str) -> str:
    """Wrap standalone display equations in ``$$`` fences.

    This does not recover LaTeX -- the PDF never stored it. It delimits the maths so a
    chunker will not split a formula in half and a renderer can typeset it. For real
    LaTeX, convert from the arXiv source, or try `--converter docling`, which runs a
    formula model.
    """
    out, in_fence = [], False
    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if not in_fence and looks_like_display_equation(line):
            out.append(f"$$ {line.strip()} $$")
        else:
            out.append(line)
    return "\n".join(out)


# --- HTML tables -----------------------------------------------------------------------
_HTML_TABLE = re.compile(r"<table\b.*?</table>", re.I | re.S)
_HTML_ROW = re.compile(r"<tr\b.*?</tr>", re.I | re.S)
_HTML_CELL = re.compile(r"<t[hd]\b([^>]*)>(.*?)</t[hd]>", re.I | re.S)
_HTML_TAG = re.compile(r"<[^>]+>")
_SPAN = re.compile(r'\b(?:col|row)span\s*=\s*["\']?(\d+)', re.I)


def _cell_text(raw: str) -> str:
    """One HTML cell as plain text, with block tags becoming spaces."""
    import html as _html

    text = re.sub(r"<br\s*/?>|</?p\b[^>]*>|</li>", " ", raw, flags=re.I)
    return " ".join(_html.unescape(_HTML_TAG.sub("", text)).split())


def html_table_to_rows(block: str) -> list[list[str]]:
    """Flatten one ``<table>`` into rectangular rows.

    A cell spanning N columns is repeated N times, which is how a reader of the pipe
    table would see it and keeps every row the same width. Row spans are not tracked
    across rows -- that needs a grid model, and the geometric backends already collapse
    the same structures, so this stays consistent with the rest of the corpus.
    """
    rows: list[list[str]] = []
    for row_html in _HTML_ROW.findall(block):
        cells: list[str] = []
        for attrs, raw in _HTML_CELL.findall(row_html):
            text = _cell_text(raw)
            span = 1
            match = _SPAN.search(attrs or "")
            if match and "colspan" in (attrs or "").lower():
                span = max(1, min(int(match.group(1)), 16))
            cells.extend([text] * span)
        if cells:
            rows.append(cells)
    return rows


def html_tables_to_pipe(markdown: str) -> str:
    """Replace every ``<table>`` in a transcription with a GitHub pipe table.

    LightOnOCR-2 emits tables as HTML on purpose -- nested tables cannot survive markdown
    -- but the rest of this corpus is pipe tables, and `lift_tables` only recognises
    those. Converting here means one backend's output is not shaped differently from
    every other paper's. A table that flattens to nothing is left as it was.
    """
    def replace(match: re.Match[str]) -> str:
        rows = html_table_to_rows(match.group(0))
        rows = [[clean_cell(c) for c in row] for row in rows]
        rows = [r for r in rows if any(r)]
        if len(rows) < 2:
            return match.group(0)
        return "\n\n" + render_rows(rows) + "\n\n"

    return _HTML_TABLE.sub(replace, markdown)


@dataclass
class TableBlock:
    index: int                  # 1-based, in document order
    page: int                   # 1-based; 0 when the backend cannot report it
    n_rows: int
    n_cols: int
    markdown: str
    caption: str = ""
    columns: list[str] = field(default_factory=list)
    kind: str = "table"         # "table" | "pseudocode"


@dataclass
class ConversionResult:
    body_markdown: str
    tables: list[TableBlock] = field(default_factory=list)
    n_pages: int = 0
    n_chars: int = 0
    truncated: bool = False

    @property
    def n_tables(self) -> int:
        return len(self.tables)


def _find_caption(lines: list[str], start: int, end: int) -> str:
    """Look for a caption immediately above or below a table block.

    Journals put captions above tables, most arXiv templates put them below, so both
    directions are searched -- nearest first.
    """
    for offset in range(1, CAPTION_LOOKAROUND + 1):
        for probe in (start - offset, end + offset - 1):
            if not (0 <= probe < len(lines)):
                continue
            text = lines[probe].strip()
            if not text or _TABLE_LINE.match(text):
                continue
            if _CAPTION.match(text) or _ALGO_CAPTION.match(text):
                return " ".join(text.split())
    return ""


def lift_tables(
    markdown: str,
    page: int,
    start_index: int,
    *,
    detect_pseudocode: bool = True,
    max_columns: int = DEFAULT_MAX_TABLE_COLUMNS,
) -> tuple[str, list[TableBlock]]:
    """Replace every pipe table in `markdown` with a ``[[TABLE:n]]`` marker.

    Shared by all backends: they all emit GitHub-style pipe tables, so pulling them out,
    cleaning them and finding their captions is one job done once rather than five times.
    """
    lines = markdown.splitlines()
    out: list[str] = []
    tables: list[TableBlock] = []
    idx = start_index
    i = 0

    while i < len(lines):
        # A table is >= 2 consecutive pipe rows whose second row is a separator.
        if (
            _TABLE_LINE.match(lines[i])
            and i + 1 < len(lines)
            and _SEP_LINE.match(lines[i + 1])
        ):
            j = i
            while j < len(lines) and _TABLE_LINE.match(lines[j]):
                j += 1

            raw = [ln for ln in lines[i:j] if not _SEP_LINE.match(ln)]
            rows = [[clean_cell(c) for c in split_row(ln)] for ln in raw]
            rows = [r for r in rows if any(r)]
            if not rows:
                i = j
                continue

            caption = _find_caption(lines, i, j)
            is_algo = detect_pseudocode and looks_like_pseudocode(rows, caption)

            # Drop noise rather than pass it downstream. The marker is not emitted
            # either, so table numbering stays contiguous and every [[TABLE:n]] in the
            # body resolves to a real section in the tables file.
            if not is_algo and is_degenerate(rows, max_columns=max_columns):
                i = j
                continue

            header = rows[0] if not is_algo else []

            tables.append(TableBlock(
                index=idx,
                page=page,
                n_rows=max(len(rows) - (0 if is_algo else 1), 0),
                n_cols=max((len(r) for r in rows), default=0),
                markdown=render_pseudocode(rows) if is_algo else render_rows(rows),
                caption=caption,
                columns=strip_placeholder_columns(header),
                kind="pseudocode" if is_algo else "table",
            ))
            out.append(MARKER.format(n=idx))
            idx += 1
            i = j
            continue
        out.append(lines[i])
        i += 1

    return "\n".join(out), tables


# Minimum share of the first pass's body text a fallback must preserve to be trusted.
FALLBACK_MIN_BODY_RATIO = 0.5


def accept_fallback(first: "ConversionResult", retry: "ConversionResult") -> bool:
    """Should a fallback-strategy retry replace the first-pass result?

    Only if it actually found tables *and* did not eat the paper doing it. Measured on
    a real 75-page arXiv paper, the "text" strategy reported 70 tables while collapsing
    the body from 252,000 characters to 3,000 -- it treats ordinary prose blocks as
    table cells. A retry that guts the body is worse than no tables at all.
    """
    if not retry.tables:
        return False
    if first.n_chars == 0:
        return True
    return retry.n_chars >= FALLBACK_MIN_BODY_RATIO * first.n_chars


class BaseConverter(ABC):
    """Contract every backend implements."""

    name = "base"

    def __init__(self, cfg: Any):
        self.cfg = cfg

    @property
    def detect_pseudocode(self) -> bool:
        return bool(getattr(self.cfg, "detect_pseudocode", True))

    @property
    def preserve_equations(self) -> bool:
        return bool(getattr(self.cfg, "preserve_equations", True))

    @property
    def max_table_columns(self) -> int:
        return int(getattr(self.cfg, "max_table_columns", DEFAULT_MAX_TABLE_COLUMNS))

    def finish(self, result: ConversionResult) -> ConversionResult:
        """Post-processing every backend shares."""
        if self.preserve_equations:
            result.body_markdown = mark_equations(result.body_markdown)
            result.n_chars = len(result.body_markdown)
        return result

    @abstractmethod
    def convert(self, pdf_path: Path) -> ConversionResult:
        ...


class PyMuPDFConverter(BaseConverter):
    """Default. Fast (tens of pages/sec), pure wheel, no Java, no ML model.

    `pymupdf4llm` handles reading order, heading detection and image suppression;
    `find_tables` (a port of pdfplumber's algorithm) recovers table structure from
    ruling lines and word positions.
    """

    name = "pymupdf"

    def __init__(self, cfg: Any):
        super().__init__(cfg)
        import pymupdf
        import pymupdf4llm

        self._pymupdf = pymupdf
        self._p4l = pymupdf4llm
        # to_markdown's keyword set moves between releases, so unsupported names are
        # filtered out. Some releases wrap it as (*args, **kwargs), which advertises no
        # names at all -- passing everything is right there, and filtering would
        # silently drop every option and quietly produce untabled, image-laden output.
        params = inspect.signature(pymupdf4llm.to_markdown).parameters
        self._accepts_any_kwarg = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        self._supported = set(params)

    def _to_markdown(self, doc: Any, pages: list[int], strategy: str) -> list[dict]:
        kwargs: dict[str, Any] = {
            "pages": pages,
            "page_chunks": True,
            # Figures are dropped by `ignore_images`. `ignore_graphics` must stay False:
            # it suppresses vector drawings, which are exactly the ruling lines
            # `find_tables` detects tables from -- turning every table into loose text.
            "ignore_images": True,
            "ignore_graphics": False,
            "table_strategy": strategy,
            "show_progress": False,
        }
        if not self._accepts_any_kwarg:
            kwargs = {k: v for k, v in kwargs.items() if k in self._supported}
        return self._p4l.to_markdown(doc, **kwargs)

    def _has_horizontal_rules(self, doc: Any, sample_pages: int = 20) -> bool:
        """Cheap proxy for a LaTeX `booktabs` table: several wide, flat rules.

        booktabs draws \\toprule/\\midrule/\\bottomrule and no vertical lines at all,
        which is exactly the case `lines_strict` tends to miss.
        """
        rules = 0
        for page in doc.pages(0, min(doc.page_count, sample_pages)):
            width = page.rect.width or 1.0
            for drawing in page.get_drawings():
                rect = drawing.get("rect")
                if rect is None:
                    continue
                if rect.height <= 2.0 and rect.width >= 0.2 * width:
                    rules += 1
                    if rules >= 3:
                        return True
        return False

    def _assemble(self, chunks: list[dict]) -> ConversionResult:
        if isinstance(chunks, str):
            # page_chunks was not honoured; fall back to treating it as one blob rather
            # than iterating the string character by character.
            chunks = [{"text": chunks, "metadata": {"page": 1}}]
        body_parts: list[str] = []
        tables: list[TableBlock] = []
        for offset, chunk in enumerate(chunks):
            page_no = (chunk.get("metadata") or {}).get("page", offset + 1)
            text, found = lift_tables(
                chunk.get("text", ""), page_no, len(tables) + 1,
                detect_pseudocode=self.detect_pseudocode,
                max_columns=self.max_table_columns,
            )
            tables.extend(found)
            body_parts.append(text)
        body = "\n\n".join(p.strip() for p in body_parts if p.strip())
        return ConversionResult(body_markdown=body, tables=tables, n_chars=len(body))

    def convert(self, pdf_path: Path) -> ConversionResult:
        doc = self._pymupdf.open(pdf_path)
        try:
            max_pages = self.cfg.max_pages
            n_pages = doc.page_count
            truncated = n_pages > max_pages
            pages = list(range(min(n_pages, max_pages)))

            result = self._assemble(self._to_markdown(doc, pages, self.cfg.table_strategy))

            # booktabs fallback: no tables found, but the page clearly has rules on it.
            # Off by default -- see accept_fallback and the note in config.yaml.
            fallback = self.cfg.table_fallback_strategy
            if not result.tables and fallback and self._has_horizontal_rules(doc):
                retry = self._assemble(self._to_markdown(doc, pages, fallback))
                if accept_fallback(result, retry):
                    result = retry

            result.n_pages = n_pages
            result.truncated = truncated
            return self.finish(result)
        finally:
            doc.close()


class PdfPlumberConverter(BaseConverter):
    """MIT-licensed alternative. Same table algorithm as PyMuPDF's, ~10-20x slower."""

    name = "pdfplumber"

    def __init__(self, cfg: Any):
        super().__init__(cfg)
        import pdfplumber

        self._pdfplumber = pdfplumber

    def convert(self, pdf_path: Path) -> ConversionResult:
        tables: list[TableBlock] = []
        body_parts: list[str] = []
        with self._pdfplumber.open(pdf_path) as pdf:
            n_pages = len(pdf.pages)
            for page_no, page in enumerate(pdf.pages[: self.cfg.max_pages], start=1):
                found = page.find_tables()
                boxes = [t.bbox for t in found]

                # Body text = everything not sitting inside a detected table.
                def outside(obj: dict, boxes: list[tuple] = boxes) -> bool:
                    cx = (obj["x0"] + obj["x1"]) / 2
                    cy = (obj["top"] + obj["bottom"]) / 2
                    return not any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in boxes)

                text = (page.filter(outside).extract_text() or "") if boxes else (page.extract_text() or "")

                for t in found:
                    rows = [[clean_cell(c or "") for c in row] for row in t.extract()]
                    rows = [r for r in rows if any(r)]
                    if not rows:
                        continue
                    caption = self._caption_near(page, t)
                    is_algo = self.detect_pseudocode and looks_like_pseudocode(rows, caption)
                    if not is_algo and is_degenerate(rows, max_columns=self.max_table_columns):
                        continue
                    idx = len(tables) + 1
                    tables.append(TableBlock(
                        index=idx, page=page_no,
                        n_rows=max(len(rows) - (0 if is_algo else 1), 0),
                        n_cols=max((len(r) for r in rows), default=0),
                        markdown=render_pseudocode(rows) if is_algo else render_rows(rows),
                        caption=caption,
                        columns=strip_placeholder_columns(rows[0] if not is_algo else []),
                        kind="pseudocode" if is_algo else "table",
                    ))
                    text = f"{text}\n\n{MARKER.format(n=idx)}"
                body_parts.append(text)

        body = "\n\n".join(p.strip() for p in body_parts if p.strip())
        return self.finish(ConversionResult(
            body_markdown=body, tables=tables,
            n_pages=n_pages, n_chars=len(body),
            truncated=n_pages > self.cfg.max_pages,
        ))

    @staticmethod
    def _caption_near(page: Any, table: Any) -> str:
        """Text just above or below the table bbox that reads like a caption."""
        x0, top, x1, bottom = table.bbox
        for band in ((max(top - 60, 0), top), (bottom, min(bottom + 60, page.height))):
            try:
                crop = page.crop((0, band[0], page.width, band[1])).extract_text() or ""
            except ValueError:
                continue
            for line in crop.splitlines():
                if _CAPTION.match(line) or _ALGO_CAPTION.match(line):
                    return " ".join(line.split())
        return ""


class DoclingConverter(BaseConverter):
    """Best table and formula fidelity (TableFormer plus a code/formula model), but
    ~1-5 s/page on CPU -- an opt-in re-run for hard papers, not a corpus-scale default.
    The right choice for equation-heavy work."""

    name = "docling"

    def __init__(self, cfg: Any):
        super().__init__(cfg)
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False # Skip OCR

        # docling's *own* deadline, and the only one that actually works here. Its
        # pipeline runs stages on background threads, so the SIGALRM backstop in
        # `convert_and_write` fires on the main thread while those threads keep grinding
        # -- measured: a 5 second alarm on 0706.3792, still burning CPU 100 seconds
        # later, and workers in the live run stuck 25 minutes against a 120 second
        # timeout. `document_timeout` is cooperative: the stages check it and unwind.
        # It defaults to None, i.e. no limit at all, which is what let those papers hang.
        timeout = getattr(cfg, "timeout", None)
        if timeout:
            pipeline_options.document_timeout = float(timeout)

        # Pin the accelerator rather than leaving it on "auto". Each worker process
        # builds its own models and its own CUDA context (~2 GB of VRAM measured), so
        # this is the knob that decides how many workers a GPU can hold -- and an
        # unnoticed fall back to CPU is an order-of-magnitude difference in throughput.
        device = getattr(cfg, "device", "auto") or "auto"
        threads = getattr(cfg, "num_threads", None)
        try:
            from docling.datamodel.accelerator_options import (
                AcceleratorDevice, AcceleratorOptions,
            )
            pipeline_options.accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice(device),
                **({"num_threads": int(threads)} if threads else {}),
            )
        except (ImportError, ValueError) as exc:
            # An older docling, or a device name it does not know: fall through to its
            # own defaults rather than refusing to convert.
            logging.getLogger(__name__).warning(
                "docling accelerator not configured (device=%r): %s", device, exc
            )

        format_option: dict[str, Any] = {"pipeline_options": pipeline_options}

        # docling's default text backend, `docling_parse`, deadlocks on some PDFs. When
        # `document_timeout` fires, its page producer is abandoned ("did not terminate
        # within 15.0s") and the main thread then blocks *forever* in `_unload()`, inside
        # the docling_parse C extension. No Python-level timeout can break that: the
        # interpreter never regains control, so SIGALRM is recorded and never delivered.
        # That is what wedged both workers of the live run at zero throughput.
        #
        # Measured on 0706.3792, one of the papers that had "failed" three times:
        #   docling_parse  hung indefinitely (killed at 25 min of CPU)
        #   pypdfium       converted in 13.1s, 252,110 characters
        # and on a well-behaved paper pypdfium was 1.8s vs 3.9s for the same text.
        # Table structure comes from TableFormer either way -- this only swaps the text
        # extractor -- so the fidelity the docling backend is chosen for is unaffected.
        if (getattr(cfg, "pdf_backend", "pypdfium") or "pypdfium") == "pypdfium":
            try:
                from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
                format_option["backend"] = PyPdfiumDocumentBackend
            except ImportError:                       # fall back to docling's default
                logging.getLogger(__name__).warning(
                    "pypdfium backend unavailable; using docling_parse, which can hang"
                )

        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(**format_option)}
        )

    def convert(self, pdf_path: Path) -> ConversionResult:
        conv_res = self._converter.convert(str(pdf_path))

        # `document_timeout` does not raise -- it stops work and hands back whatever was
        # assembled so far, flagged PARTIAL_SUCCESS. Measured on 0706.3792 with a 3s
        # timeout: 97,686 characters instead of 206,672 and **0 tables instead of 24**,
        # which without this check would have been written out and recorded `done`.
        # A silently half-converted paper is worse than a failed one: nothing downstream
        # can tell it is incomplete. Raising here marks it failed and, when a
        # `fallback_converter` is configured, hands the paper to it instead.
        status = getattr(conv_res, "status", None)
        if status is not None and getattr(status, "name", "") != "SUCCESS":
            # docling reports one error *per failed page*, so a timed-out 128-page paper
            # yields 79 copies of "document timeout exceeded" plus the one line that
            # actually says something. Deduplicated, order preserved.
            seen: dict[str, int] = {}
            for err in getattr(conv_res, "errors", None) or []:
                message = str(getattr(err, "error_message", err)).strip()
                seen[message] = seen.get(message, 0) + 1
            detail = "; ".join(
                m if n == 1 else f"{m} (x{n})" for m, n in seen.items()
            )
            raise RuntimeError(
                f"docling returned {getattr(status, 'name', status)}"
                + (f": {detail}" if detail else " (likely document_timeout)")
            )

        doc = conv_res.document
        body, tables = lift_tables(
            doc.export_to_markdown(), page=0, start_index=1,
            detect_pseudocode=self.detect_pseudocode,
            max_columns=self.max_table_columns,
        )
        return self.finish(ConversionResult(
            body_markdown=body, tables=tables,
            n_pages=len(getattr(doc, "pages", []) or []), n_chars=len(body),
        ))


class OpenDataLoaderConverter(BaseConverter):
    """Apache-2.0, strong structural fidelity -- but shells out to a JVM and needs
    JDK 11+ on PATH."""

    name = "opendataloader"

    def __init__(self, cfg: Any):
        super().__init__(cfg)
        import opendataloader_pdf

        self._odl = opendataloader_pdf

    def convert(self, pdf_path: Path) -> ConversionResult:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self._odl.convert(input_path=[str(pdf_path)], output_dir=tmp, format="markdown")
            produced = sorted(Path(tmp).rglob("*.md"))
            if not produced:
                raise RuntimeError("opendataloader produced no markdown output")
            raw = produced[0].read_text(encoding="utf-8", errors="replace")
        body, tables = lift_tables(
            raw, page=0, start_index=1, detect_pseudocode=self.detect_pseudocode,
            max_columns=self.max_table_columns,
        )
        return self.finish(ConversionResult(body_markdown=body, tables=tables, n_chars=len(body)))


class LightOnOCRConverter(BaseConverter):
    """`lightonai/LightOnOCR-2-1B` -- an end-to-end OCR vision model.

    The reason to reach for it is the one gap the geometric backends cannot close:
    equations. Every other converter here recovers the PDF's *text layer*, so a formula
    arrives as its visual approximation and `mark_equations` can only fence it. This
    model was distilled on transcriptions that carry real LaTeX spans, with arXiv well
    represented, so the maths comes back as maths. It also reads scans, which is what
    the `low_text` flag exists to mark.

    The cost is severe. It renders every page to an image and generates tokens for it:
    LightOn measure 5.71 pages/s on an H100, and a mid-range card is a fraction of that
    against ~0.12 s/page for docling. This is a backend for one paper, or for the scanned
    and equation-heavy minority -- not for a 2.8M-paper crawl.

    Tables come back as HTML by design ("some nested tables cannot be represented in
    markdown"), so they are converted to pipe tables here to match the rest of the corpus.
    """

    name = "lightonocr"
    model_id = "lightonai/LightOnOCR-2-1B"
    # LightOn's stated preprocessing: 200 DPI, longest side 1540px, aspect preserved.
    render_dpi = 200
    longest_side = 1540

    def __init__(self, cfg: Any):
        super().__init__(cfg)
        import pypdfium2
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self._pdfium = pypdfium2
        self._torch = torch

        model_id = getattr(cfg, "lightonocr_model", None) or self.model_id
        device = getattr(cfg, "device", "auto") or "auto"
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForVision2Seq.from_pretrained(
            model_id, dtype=dtype, device_map=device if device == "cuda" else None,
        )
        if device != "cuda":
            self._model.to(device)
        self._model.eval()
        self._max_new_tokens = int(getattr(cfg, "lightonocr_max_tokens", 4096))

    def _render(self, pdf_path: Path, max_pages: int) -> list[Any]:
        """PDF pages as PIL images at the resolution the model was trained on."""
        doc = self._pdfium.PdfDocument(str(pdf_path))
        try:
            n_pages = len(doc)
            images = []
            for index in range(min(n_pages, max_pages)):
                page = doc[index]
                # pypdfium's scale is relative to 72 dpi.
                bitmap = page.render(scale=self.render_dpi / 72)
                image = bitmap.to_pil()
                longest = max(image.size)
                if longest > self.longest_side:
                    ratio = self.longest_side / longest
                    image = image.resize(
                        (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
                    )
                images.append(image)
            return images, n_pages
        finally:
            doc.close()

    def _transcribe(self, image: Any) -> str:
        messages = [{"role": "user", "content": [{"type": "image"}]}]
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens, do_sample=False,
            )
        # Drop the prompt tokens; only the continuation is the transcription.
        start = inputs["input_ids"].shape[-1]
        return self._processor.decode(generated[0][start:], skip_special_tokens=True)

    def convert(self, pdf_path: Path) -> ConversionResult:
        images, n_pages = self._render(pdf_path, self.cfg.max_pages)
        body_parts: list[str] = []
        tables: list[TableBlock] = []

        for page_no, image in enumerate(images, start=1):
            markdown = html_tables_to_pipe(self._transcribe(image))
            text, found = lift_tables(
                markdown, page_no, len(tables) + 1,
                detect_pseudocode=self.detect_pseudocode,
                max_columns=self.max_table_columns,
            )
            tables.extend(found)
            body_parts.append(text)

        body = "\n\n".join(p.strip() for p in body_parts if p.strip())
        return self.finish(ConversionResult(
            body_markdown=body, tables=tables, n_pages=n_pages, n_chars=len(body),
            truncated=n_pages > self.cfg.max_pages,
        ))


class MarkItDownConverter(BaseConverter):
    """Microsoft's markitdown. Fast and dependency-light, but its PDF path is a plain
    pdfminer text dump: no table structure at all. Useful mainly as a baseline in the
    parser comparison -- if it matches a structured backend, the page was simple."""

    name = "markitdown"

    def __init__(self, cfg: Any):
        super().__init__(cfg)
        from markitdown import MarkItDown

        self._md = MarkItDown()

    def convert(self, pdf_path: Path) -> ConversionResult:
        raw = self._md.convert(str(pdf_path)).text_content or ""
        body, tables = lift_tables(
            raw, page=0, start_index=1, detect_pseudocode=self.detect_pseudocode,
            max_columns=self.max_table_columns,
        )
        return self.finish(ConversionResult(body_markdown=body, tables=tables, n_chars=len(body)))


REGISTRY: dict[str, type[BaseConverter]] = {
    PyMuPDFConverter.name: PyMuPDFConverter,
    PdfPlumberConverter.name: PdfPlumberConverter,
    DoclingConverter.name: DoclingConverter,
    OpenDataLoaderConverter.name: OpenDataLoaderConverter,
    LightOnOCRConverter.name: LightOnOCRConverter,
    MarkItDownConverter.name: MarkItDownConverter,
}


def get_converter(name: str, cfg: Any) -> BaseConverter:
    try:
        cls = REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown converter {name!r}; choose one of {sorted(REGISTRY)}"
        ) from None
    return cls(cfg)


# --- process-pool entry point --------------------------------------------------------
# Conversion runs in a *process* pool: PyMuPDF is a C extension, so a segfault on a
# malformed PDF costs one worker rather than the whole run. The worker writes the output
# files itself and returns only a small TaskResult, so no large markdown string is ever
# pickled back across the process boundary.

_WORKER_CACHE: dict[str, BaseConverter] = {}
_STORE_CACHE: dict[str, Any] = {}


def _worker_uploader(minio_cfg: Any, arxiv_id: str) -> Any:
    """An `upload(kind, path)` callback for one paper, or None when storage is local.

    One MinIO client per worker process, built on first use and reused -- the same shape
    as the converter cache, and for the same reason: a connection per paper would be
    absurd, and a connection created in the parent could not cross into a spawned worker.
    """
    if not minio_cfg:
        return None
    from .objectstore import MinioSettings, MinioStore, upload_and_unlink

    store = _STORE_CACHE.get("minio")
    if store is None:
        store = _STORE_CACHE["minio"] = MinioStore(MinioSettings(**minio_cfg))

    def upload(kind: str, path: Path) -> None:
        upload_and_unlink(store, path, store.name_for(kind, arxiv_id))

    return upload


def _worker_converter(name: str, cfg: Any) -> BaseConverter:
    """One converter instance per worker process, built lazily and reused."""
    conv = _WORKER_CACHE.get(name)
    if conv is None:
        conv = _WORKER_CACHE[name] = get_converter(name, cfg)
    return conv


# How often the timeout re-fires once the deadline has passed. Belt-and-braces: the
# first one should escape now, but a swallowed timeout used to mean an unbounded run.
TIMEOUT_REPEAT_INTERVAL = 5.0

# The SIGALRM backstop deliberately trails the backend's own deadline. docling shuts its
# stages down cooperatively (`document_timeout`, then `stage_shutdown_timeout_seconds`,
# 15s by default); interrupting mid-unwind would turn a clean, explicable failure into a
# torn one. These leave room for that and still bound the worst case.
BACKSTOP_TIMEOUT_FACTOR = 1.5
BACKSTOP_TIMEOUT_GRACE = 30.0


class ConversionTimeout(BaseException):
    """Deliberately **not** an `Exception`.

    The timeout is raised by a SIGALRM handler part-way down a backend's own call stack,
    and every backend wraps its work in a broad `except Exception`. docling's ends with

        except Exception as e:
            raise RuntimeError(f"Pipeline {self.__class__.__name__} failed") from e

    so a timeout derived from `Exception` was caught, re-raised as an anonymous pipeline
    error, and -- because a one-shot `signal.alarm` had already been spent -- the
    conversion then ran on with no deadline at all. Measured on 0706.3792: a **5 second**
    alarm, still burning CPU two minutes later; in the live run, workers grinding single
    papers for 25 minutes against a 120 second timeout.

    Inheriting from `BaseException` puts it past every `except Exception` between here
    and the handler, which is the right semantics anyway: abandoning the document is
    control flow, not an error the backend is invited to handle.
    """


def describe_exception(exc: BaseException, *, limit: int = 4) -> str:
    """Flatten an exception and the chain it was raised from into one line.

    Backends wrap failures and lose the diagnosis. docling's pipeline ends with

        raise RuntimeError(f"Pipeline {self.__class__.__name__} failed") from e

    and logs nothing, so recording only the outermost message leaves every failure
    reading "RuntimeError: Pipeline StandardPdfPipeline failed" -- true, and useless.
    The `from e` cause is where the real error lives, so it is walked and joined.

    `__context__` is followed only when the raise did not suppress it, which keeps
    incidental "during handling of the above exception" noise out.
    """
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and len(parts) < limit and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}".strip())
        nxt = current.__cause__
        if nxt is None and not current.__suppress_context__:
            nxt = current.__context__
        current = nxt
    return " <- caused by ".join(parts)


def _converter_chain(convert_cfg: Any) -> list[tuple[str, bool]]:
    """The backends to try, in order, each flagged with whether it is the last chance.

    A fallback earns its place because the two backends fail on different things: the
    model pipeline is the one that times out on long or awkward documents, and pymupdf
    is ~100x faster and has no model, no GPU and no threads to deadlock in. A paper the
    good backend cannot manage is better served by a plainer conversion than by nothing.
    """
    chain = [convert_cfg.converter]
    fallback = getattr(convert_cfg, "fallback_converter", None)
    if fallback and fallback not in chain:
        chain.append(fallback)
    return [(name, i == len(chain) - 1) for i, name in enumerate(chain)]


def _convert_with_deadline(name: str, pdf_path: Path, convert_cfg: Any) -> ConversionResult:
    """Run one backend under the SIGALRM backstop, then disarm it."""
    import signal
    import threading

    # The backstop fires later than the backend's own deadline, so a backend that can
    # unwind cleanly (docling, via `document_timeout`) gets the chance to do so and
    # report a real error. The signal is the fallback for backends that cannot.
    deadline = convert_cfg.timeout * BACKSTOP_TIMEOUT_FACTOR + BACKSTOP_TIMEOUT_GRACE

    def _timeout(_signum: int, _frame: Any) -> None:
        raise ConversionTimeout(
            f"{name} exceeded {convert_cfg.timeout}s "
            f"(backstop fired at {deadline:.0f}s)"
        )

    # `signal.signal` raises outright off the main thread, which would turn the backstop
    # from a safety net into the thing that fails the conversion. The production path is
    # a process pool, so this is armed there; a threaded caller simply relies on the
    # backend's own deadline instead.
    armed = (
        hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )
    if armed:
        signal.signal(signal.SIGALRM, _timeout)
        # A repeating timer, not a one-shot alarm. If anything downstream still contrives
        # to swallow the first timeout, the next arrives seconds later rather than never.
        signal.setitimer(signal.ITIMER_REAL, deadline, TIMEOUT_REPEAT_INTERVAL)
    try:
        return _worker_converter(name, convert_cfg).convert(pdf_path)
    finally:
        if armed:
            signal.setitimer(signal.ITIMER_REAL, 0)


def convert_and_write(
    row: Any,
    pdf_path: Path,
    data_dir: Path,
    convert_cfg: Any,
    *,
    base_url: str,
    pdf_bytes: int | None = None,
    pdf_sha256: str | None = None,
    keep_pdf: bool = False,
    minio: dict[str, Any] | None = None,
) -> Any:
    """Convert one staged PDF, write the outputs, drop the PDF. Returns a `TaskResult`.

    Tries `convert.converter`, then `convert.fallback_converter` if the first one fails.
    The PDF is deleted once, after every attempt, so the fallback still has an input.
    """
    import os

    worker_id = os.getpid()

    from .state import DONE, FAILED_CONVERT, TaskResult
    from .writer import write_outputs

    log_ = logging.getLogger(__name__)
    errors: list[str] = []

    try:
        for name, is_last in _converter_chain(convert_cfg):
            try:
                result = _convert_with_deadline(name, pdf_path, convert_cfg)

                # Backends that do not report a page count (markitdown) make this ratio
                # meaningless, so the flag is simply not raised for them.
                low_text = bool(
                    result.n_pages
                    and result.n_chars / result.n_pages < convert_cfg.min_chars_per_page
                )
                md_bytes, tables_bytes = write_outputs(
                    data_dir, row, result,
                    # The backend that actually produced this paper, which is not
                    # necessarily the configured one. It lands in the markdown front
                    # matter, so "which papers took the fallback" stays answerable.
                    converter=name,
                    base_url=base_url,
                    upload=_worker_uploader(minio, row.arxiv_id),
                )
                if errors:
                    log_.warning("%s converted by fallback %s after: %s",
                                 row.arxiv_id, name, " | ".join(errors))
                return TaskResult(
                    arxiv_id=row.arxiv_id, status=DONE,
                    pdf_bytes=pdf_bytes, pdf_sha256=pdf_sha256,
                    md_bytes=md_bytes, tables_bytes=tables_bytes,
                    n_pages=result.n_pages, n_tables=result.n_tables,
                    n_chars=result.n_chars, low_text=low_text,
                    # The upload removes the local copy, so `verify` must not go looking
                    # for it on disk -- without this flag it reports every paper of a
                    # run-minio corpus as missing, and `--fix` re-queues the lot.
                    remote_only=bool(minio),
                    count_attempt=True, worker_id=worker_id, converter=name,
                )

            except BaseException as exc:  # noqa: BLE001 - recorded, never raised
                # The one-line summary goes to the manifest and the terminal; the full
                # traceback goes to the log file, the only place with room for it.
                log_.error("conversion failed: %s via %s", row.arxiv_id, name, exc_info=exc)
                errors.append(f"{name}: {describe_exception(exc)}")
                if isinstance(exc, ConversionTimeout):
                    # The backend was interrupted mid-call, so whatever state it holds --
                    # a CUDA context, a half-drained page queue -- cannot be trusted for
                    # the next paper. Dropping the cached instance costs one model reload
                    # and contains the damage instead of poisoning the worker's queue.
                    _WORKER_CACHE.pop(name, None)
                if is_last:
                    return TaskResult(
                        arxiv_id=row.arxiv_id, status=FAILED_CONVERT,
                        error=" | then ".join(errors)[:500],
                        pdf_bytes=pdf_bytes, pdf_sha256=pdf_sha256,
                        count_attempt=True, worker_id=worker_id,
                    )
    finally:
        if not keep_pdf:
            pdf_path.unlink(missing_ok=True)
        # Give the heap back between papers. Without this a worker's RSS climbs about
        # 0.3 GB per document and never comes down -- see memory.release_memory.
        from .memory import release_memory

        release_memory()
