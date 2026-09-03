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
        from docling.document_converter import DocumentConverter

        self._converter = DocumentConverter()

    def convert(self, pdf_path: Path) -> ConversionResult:
        doc = self._converter.convert(str(pdf_path)).document
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


def _worker_converter(name: str, cfg: Any) -> BaseConverter:
    """One converter instance per worker process, built lazily and reused."""
    conv = _WORKER_CACHE.get(name)
    if conv is None:
        conv = _WORKER_CACHE[name] = get_converter(name, cfg)
    return conv


class ConversionTimeout(RuntimeError):
    pass


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
) -> Any:
    """Convert one staged PDF, write the outputs, drop the PDF. Returns a `TaskResult`."""
    import os
    import signal

    worker_id = os.getpid()

    from .state import DONE, FAILED_CONVERT, TaskResult
    from .writer import write_outputs

    def _timeout(_signum: int, _frame: Any) -> None:
        raise ConversionTimeout(f"conversion exceeded {convert_cfg.timeout}s")

    armed = hasattr(signal, "SIGALRM")
    if armed:
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(int(convert_cfg.timeout))

    try:
        converter = _worker_converter(convert_cfg.converter, convert_cfg)
        result = converter.convert(pdf_path)

        # Backends that do not report a page count (markitdown) make this ratio
        # meaningless, so the flag is simply not raised for them.
        low_text = bool(
            result.n_pages
            and result.n_chars / result.n_pages < convert_cfg.min_chars_per_page
        )

        md_bytes, tables_bytes = write_outputs(
            data_dir, row, result,
            converter=convert_cfg.converter,
            base_url=base_url,
        )
        return TaskResult(
            arxiv_id=row.arxiv_id, status=DONE,
            pdf_bytes=pdf_bytes, pdf_sha256=pdf_sha256,
            md_bytes=md_bytes, tables_bytes=tables_bytes,
            n_pages=result.n_pages, n_tables=result.n_tables,
            n_chars=result.n_chars, low_text=low_text,
            count_attempt=True, worker_id=worker_id,
        )
    except BaseException as exc:  # noqa: BLE001 - any failure must be recorded, not raised
        return TaskResult(
            arxiv_id=row.arxiv_id, status=FAILED_CONVERT,
            error=f"{type(exc).__name__}: {exc}"[:500],
            pdf_bytes=pdf_bytes, pdf_sha256=pdf_sha256,
            count_attempt=True, worker_id=worker_id,
        )
    finally:
        if armed:
            signal.alarm(0)
        if not keep_pdf:
            pdf_path.unlink(missing_ok=True)
