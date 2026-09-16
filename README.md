# arxiv-crawler

Turns the [Kaggle arXiv metadata snapshot](https://www.kaggle.com/datasets/Cornell-University/arxiv)
into a local, queryable text corpus.

For every paper it downloads the PDF from arXiv with `requests`, converts it to Markdown —
**figures dropped, tables preserved** — writes the body and the tables as separate `.md` files
plus a metadata `.json`, and then **deletes the PDF**. Download and conversion run in parallel,
and the whole thing is resumable: interrupt it at any point and start it again.

```
data/md/2301/2301.12345.md              the paper, as Markdown
data/tables/2301/2301.12345.tables.md   its tables, as Markdown pipe tables
data/meta/2301/2301.12345.json          title, authors, date, doi, categories, ...
```

---

## Contents

- [How it works](#how-it-works)
- [Quickstart](#quickstart)
- [Output format](#output-format)
- [Command reference](#command-reference)
- [Configuration](#configuration)
- [Retries](#retries)
- [Terminal output](#terminal-output)
- [Converter backends](#converter-backends)
- [Optional: Postgres catalog](#optional-postgres-catalog)
- [arXiv usage policy](#arxiv-usage-policy)
- [Scale: time and disk](#scale-time-and-disk)
- [Resuming and troubleshooting](#resuming-and-troubleshooting)
- [Known limitations](#known-limitations)
- [Development](#development)

---

## How it works

```
                    ┌──────────── data/manifest.db (SQLite, WAL) ────────────┐
                    │            one writer thread, results queue            │
                    └──────────▲──────────────────────────▲──────────────────┘
                               │                          │
  metadata JSONL ──► prepare ──┘   [pending rows]         │
                                        │                 │
                          ThreadPoolExecutor(N)   ProcessPoolExecutor(M)
                          download → data/tmp/ ──► convert → md/ tables/ meta/
                            (global token bucket)      └─► delete the PDF
                                     bounded queue
```

Three ideas carry the design:

**Two pools, because the work has two shapes.** Downloading is IO-bound and rate-capped, so it
runs on threads. PDF parsing is CPU-bound inside a C extension that can segfault on a malformed
file, so it runs on processes — a crash costs one worker, not the run.

**One writer.** Every manifest mutation goes through a single `ManifestWriter` thread draining a
queue. That sidesteps SQLite lock contention rather than fighting it with retries, and it means
the manifest is consistent no matter when you interrupt.

**The manifest is the source of truth.** `data/manifest.db` records the state of every paper.
Resuming is just "claim the rows that aren't done yet", and `verify` cross-checks it against what
is really on disk.

---

## Quickstart

### 1. Environment

Python 3.10+. Use a dedicated virtualenv.

```bash
python3 -m venv .venv && .venv/bin/python -m pip install --upgrade pip
```

```bash
.venv/bin/python -m pip install -r requirements.txt
```

> If `pip` appears to hang for minutes with no output, it is almost certainly the macOS keyring
> lookup, not the network. Re-run with `PIP_KEYRING_PROVIDER=disabled`.

### 2. Get the metadata snapshot

~4.5 GB of JSON Lines, ~2.8M records. Needs a [Kaggle API token](https://www.kaggle.com/docs/api).

```bash
kaggle datasets download -d Cornell-University/arxiv -p data/metadata --unzip
```

### 3. Look before you crawl

```bash
.venv/bin/jupyter lab notebooks/01_metadata_eda.ipynb
```

[`notebooks/01_metadata_eda.ipynb`](notebooks/01_metadata_eda.ipynb) profiles the snapshot —
papers per category, growth over time, cross-listing, versions, licenses — and ends with a cell
that turns a chosen scope into a concrete estimate of papers, hours and gigabytes. It streams the
file and caches its aggregates, so the expensive pass happens once.

### 4. Load the manifest, then run

```bash
.venv/bin/python -m src.main prepare --categories cs.CL --from 2024-01 --limit 20
```

```bash
.venv/bin/python -m src.main run
```

```bash
.venv/bin/python -m src.main status
```

Start with a small `--limit`. See [arXiv usage policy](#arxiv-usage-policy) before a large run.

---

## Output format

Three files per paper, all sharded by `yymm` so no directory grows unmanageable. Old-style IDs
are made filesystem-safe (`hep-th/9901001` → `hep-th_9901001`).

### `data/md/<shard>/<id>.md` — the paper

YAML front matter, then the body in reading order. Each table is lifted out and replaced by a
marker linking to its entry in the tables file, so you never lose where a table sat in the text.

```markdown
---
id: '1706.03762'
version: v7
title: Attention Is All You Need
authors:
- Vaswani, Ashish
- Shazeer, Noam
date_released: '2017-06-12'
doi: 10.5555/3295222.3295349
categories:
- cs.CL
- cs.LG
primary_category: cs.CL
n_pages: 15
n_tables: 2
converter: pymupdf
---

# Attention Is All You Need

The dominant sequence transduction models are based on ...

[[TABLE:1]](../../tables/1706/1706.03762.tables.md#table-1)

We trained on the standard WMT 2014 English-German dataset ...
```

### `data/tables/<shard>/<id>.tables.md` — the tables

Written only when the paper has tables. Headings match the anchors linked from the body.

```markdown
# Tables — 1706.03762

## Table 2
*page 10 · 5 rows × 3 columns*

|Parser|Training|WSJ 23 F1|
|---|---|---|
|Transformer (4 layers)|WSJ only, discriminative|91.3|
|Transformer (4 layers)|semi-supervised|92.7|
```

### `data/meta/<shard>/<id>.json` — the metadata

```json
{
  "id": "1810.04805",
  "title": "BERT: Pre-training of Deep Bidirectional Transformers ...",
  "authors": ["Devlin, Jacob", "Chang, Ming-Wei"],
  "date_released": "2018-10-11",
  "date_updated": "2019-05-24",
  "doi": null,
  "categories": ["cs.CL", "cs.LG"],
  "primary_category": "cs.CL",
  "source_url": "https://export.arxiv.org/abs/1810.04805v2",
  "n_pages": 16,
  "n_tables": 3,
  "n_chars": 65775,
  "md_path": "md/1810/1810.04805.md",
  "tables_path": "tables/1810/1810.04805.tables.md"
}
```

`date_released` is v1's submission date; `date_updated` is the latest version's.
`tables_path` is `null` when the paper has no tables.

That field list is exhaustive and enforced — `build_metadata` asserts against
`writer.METADATA_FIELDS`, so the set cannot drift silently. Everything else stays in the
manifest rather than being copied into 2.8M files: crawl bookkeeping (`attempts`, `error`,
`status`), the size and checksum of the discarded PDF (`pdf_bytes`, `pdf_sha256`), and the
per-conversion flags (`truncated`, `low_text`, `converter`). Query those with `status`,
`compare_size`, or SQL against `manifest.db`.

The snapshot's `abstract`, `journal-ref` and `license` are **not carried at all** — not into
the manifest, the metadata JSON, or the Postgres schema. If you need abstracts, they are in
the Kaggle snapshot, keyed by the same `id`.

Every file is written to a temporary name and then `os.replace`d into place, so an interrupted run
leaves either the old file or the new one — never a half-written file a resumed run would mistake
for complete.

---

## Command reference

```bash
python -m src.main prepare [--metadata PATH] [--categories cs.LG,cs.CL] [--primary-only]
                           [--from YYYY-MM] [--to YYYY-MM] [--limit N]
```
Streams the snapshot into the manifest. Idempotent — re-running with a wider filter adds work and
leaves completed papers untouched. Pins each paper's highest version, so downloads are
reproducible rather than tracking a moving `/pdf/<id>`.

```bash
python -m src.main run [--download-workers N] [--convert-workers M] [--rps R] [--burst B]
                       [--converter NAME] [--limit N] [--keep-pdf]
                       [--no-retry-failed] [--max-attempts N] [--worker-bars]
```
Downloads and converts in parallel. `Ctrl-C` once to stop cleanly (in-flight work finishes, staged
PDFs are cleared, the manifest is left consistent); twice to abort.

Every run **opens with the papers an earlier run failed on**, ahead of any fresh work, so a
transient failure heals by itself — see [Retries](#retries). `--worker-bars` adds one progress
line per conversion worker under the main bar.

```bash
python -m src.main run-minio [same flags as run]
python -m src.main dump [--keep-local] [--skip-existing] [--limit N] [--dry-run]
python -m src.main test-paper <id> [--converter NAME] [--minio] [--show N]
```
Object storage. `run-minio` is `run` with the output sent to MinIO instead of kept on
disk; `dump` migrates a corpus that is already local; `test-paper` puts one paper through
the whole path without touching the manifest. See [Object storage](#object-storage).

```bash
python -m src.main status     # counts by status, output size, tables extracted
python -m src.main verify     # cross-check the manifest against files on disk; --fix re-queues
python -m src.main retry --stage {download,convert,all} [--max-attempts N]
```

```bash
python -m src.scripts.compare_size [--csv report.csv] [--top N]
```
PDF-versus-Markdown size report, per-category breakdown, and extrapolation to the full corpus.
Reads sizes recorded in the manifest, so it works even though the PDFs are gone.

---

## Configuration

All defaults live in [`config.yaml`](config.yaml); CLI flags override them per field.

| Key | Default | Notes |
|---|---|---|
| `paths.data_dir` | `data` | Everything generated lives here, and it is gitignored. |
| `scope.categories` | `[]` | Matches **any** of a paper's categories unless `primary_only`. |
| `scope.date_from` / `date_to` | `null` | Inclusive `YYYY-MM`, on the v1 submission date. |
| `crawl.base_url` | `https://export.arxiv.org` | arXiv's host for programmatic access. |
| `crawl.contact` | placeholder | **Set this.** arXiv asks automated clients to identify themselves. |
| `crawl.rate_per_sec` | `1.0` | Global ceiling, shared by every worker. |
| `crawl.burst` | `4` | Token bucket depth. |
| `crawl.workers` | `4` | Download threads. |
| `convert.workers` | `8` | Conversion processes. |
| `convert.timeout` | `120` | Seconds per PDF; enforced with `SIGALRM` inside the worker. |
| `convert.max_pages` | `300` | Longer documents are truncated, not failed. |
| `convert.table_strategy` | `lines_strict` | See [Converter backends](#converter-backends). |
| `convert.table_fallback_strategy` | `null` | Leave off — see the note in `config.yaml`. |
| `convert.min_chars_per_page` | `100` | Below this a paper is flagged `low_text`. |
| `retry.on_start` | `true` | Re-attempt earlier failures at the start of every `run`. |
| `retry.max_attempts` | `4` | Total tries a paper ever gets before `run` stops picking it up. |

Raising `crawl.workers` increases concurrency, **never** the request rate past `rate_per_sec`.

---

## Retries

A paper that fails is not abandoned. Each `run` begins by re-attempting every `failed_download`
and `failed_convert` row still under `retry.max_attempts`, and only then moves on to `pending`
work. Retries go first because they are the smaller, more informative set: if the last run failed
because the converter's dependency was missing, you find out in the first few seconds rather than
after another hour of new downloads.

Three properties make that safe to leave on:

- **One attempt per paper per run.** The retry worklist is snapshotted before dispatch begins, so
  a paper that fails again mid-run is not immediately picked up and retried inside the same run.
- **`attempts` is always incremented**, including when a conversion worker dies outright. A PDF
  that segfaults the C extension therefore exhausts its attempts and stops being re-downloaded,
  instead of crashing a worker on every run forever.
- **Terminal failures are excluded.** `no_pdf` (404: withdrawn or source-only) is a fact, not an
  error, and is never retried.

Once a paper is out of attempts, `run` says so and leaves it alone. `retry --max-attempts N`
raises the ceiling and re-queues it explicitly, and `--no-retry-failed` skips the pass entirely
for a run that should only chew through fresh work.

> A `failed_convert` retry **re-downloads**, because the PDF is deleted after every attempt. That
> costs a request against the rate limit, so the attempt ceiling is doing real work.

---

## Terminal output

`run` prints a progress bar and nothing else. Anything worth keeping goes to
`data/logs/crawler.log`:

```
retrying 29 previously failed paper(s) first
crawl+convert:  38%|███▊      | 18,904/49,141 [2:14:07<3:34:19, 2.35paper/s, ok=18,871, fail=33, w=8]
  ✗ 0802.2167  ConversionTimeout: conversion exceeded 120s
```

Failures are the one exception — the id and its error print above the bar as they happen, written
through `tqdm.write` so they cannot corrupt it. Everything else is suppressed, including the
per-paper INFO chatter from `docling` that otherwise runs to four lines per paper in every worker
process (one observed log reached 67 MB). Third-party loggers are capped at `WARNING` in the log
file and silenced entirely in conversion workers.

`-v` puts the full stream back on stderr when you actually want to watch it.

---

## Object storage

Converted papers can live in a MinIO bucket instead of on local disk. The bucket mirrors
the local layout, so an object name follows from an arXiv id with no lookup:

```
arxiv/md/2301/2301.12345.md
arxiv/tables/2301/2301.12345.tables.md
arxiv/meta/2301/2301.12345.json
```

**Credentials come from the environment.** `config.yaml` is tracked in git and is the
wrong place for a secret key; setting them there still works but logs a warning.

```bash
export MINIO_ACCESS_KEY=...  MINIO_SECRET_KEY=...
```

```bash
python -m src.main dump --dry-run     # what would be sent, connecting to nothing
python -m src.main dump               # upload everything local, then remove it
python -m src.main run-minio          # crawl straight into the bucket
```

**Writing is local-then-upload-then-delete**, not straight to the network. The atomic
local write is what makes an interrupted run safe, and it stays until the upload has
returned — so a dropped connection leaves the paper on disk, where the next `dump` finds
it. Only a confirmed upload removes it, which also makes `dump` resumable: whatever is
still on disk is precisely what still needs sending.

`--keep-local` turns `dump` into a copy rather than a move. `--skip-existing` avoids
re-sending objects already in the bucket, at one HEAD request per file.

---

## Converter backends

The converter is a pluggable interface; each backend imports its dependency lazily, so a missing
package or absent JVM only matters if you actually select it.

| `--converter` | Speed | License | Extra requirements | Notes |
|---|---|---|---|---|
| `pymupdf` *(default)* | ~10–30 pages/s | AGPL-3.0 | — | `pymupdf4llm` for layout, `find_tables` for tables. |
| `pdfplumber` | ~1–3 pages/s | MIT | — | The licence escape hatch. Same table algorithm, far slower. |
| `docling` | ~0.2–1 pages/s | MIT | `docling` | Best table fidelity (TableFormer). Realistically an opt-in re-run, not a corpus-scale default. |
| `lightonocr` | ~1–2 pages/s* | Apache-2.0 | `transformers`, GPU, ~2 GB weights | **The only backend that recovers real LaTeX.** See below. |
| `opendataloader` | moderate | Apache-2.0 | **JDK 11+** | Strong structure, but shells out to a JVM. |

\* LightOn measure 5.71 pages/s on an H100; a mid-range card is a fraction of that.

### LightOnOCR: the one backend that reads equations

Every other converter recovers the PDF's *text layer*, so a formula arrives as its visual
approximation and `mark_equations` can only put `$$` around it.
[`lightonai/LightOnOCR-2-1B`](https://huggingface.co/lightonai/LightOnOCR-2-1B) is an
end-to-end vision model distilled on transcriptions that carry real **LaTeX spans**, with
arXiv well represented in its training corpus. It also reads scans, which is what the
`low_text` flag exists to mark.

```bash
python -m src.main test-paper 1706.03762 --converter lightonocr --show
```

Two things to know before reaching for it:

- **It is not a corpus-scale backend.** It renders every page to an image at 200 DPI and
  generates tokens for it. Against docling's ~0.12 s/page, a 2.8M-paper crawl is out of
  reach on one GPU. Use it for a single paper, for the `low_text` scans, or for
  equation-heavy work.
- **Its tables arrive as HTML**, deliberately — "some nested tables cannot be represented
  in markdown". They are converted to pipe tables on the way in, so its papers are shaped
  like every other paper in `data/tables`. Nested tables flatten, exactly as the geometric
  backends already flatten them.

---

### Why PyMuPDF, and why no AI agent for tables

`page.find_tables()` recovers table structure geometrically, from ruling lines and word positions.
It is deterministic, reproducible, needs no model, and is roughly a thousand times cheaper than
sending 2.8M papers through an LLM. On the smoke-test sample it extracted **40 tables from 7
papers**, including 17 from the ResNet paper.

> **Licensing.** PyMuPDF is AGPL-3.0. Fine for private research; if you redistribute a service
> built on this, either buy a commercial licence or switch to `--converter pdfplumber`.

### Two measured findings baked into the defaults

**`ignore_graphics` must stay off.** It suppresses vector drawings — which are exactly the ruling
lines `find_tables` detects tables from. Turning it on silently converts every table into loose
text while still looking like a successful conversion.

**The `text` table strategy is not a usable fallback.** It treats ordinary prose as table cells.
On a 75-page arXiv paper it reported 70 spurious tables while collapsing the body from 252,000
characters to 3,000, and took 250 s instead of 6.5 s. `table_fallback_strategy` therefore defaults
to `null`, and any fallback result that destroys the body is rejected outright
(`converter.accept_fallback`).

`pymupdf4llm` is pinned below `1.0` for a related reason: the 1.x line hard-imports `onnxruntime`
and runs Tesseract OCR on image-heavy pages, in every worker process. This pipeline does not want
OCR — scanned papers are flagged `low_text` for a separate pass.

---

## Optional: Postgres catalog

The per-paper JSON files are the portable source of truth. If you want to query the corpus,
[`src/scripts/ingest_postgres.py`](src/scripts/ingest_postgres.py) bulk-loads them.

```bash
.venv/bin/python -m pip install "psycopg[binary]"
```

```bash
.venv/bin/python -m src.scripts.ingest_postgres --dsn postgresql://user@localhost/arxiv
```

Rows stream through `COPY ... FROM STDIN` into a staging table and merge with
`ON CONFLICT DO UPDATE` — orders of magnitude faster than row-by-row inserts, and idempotent, so
re-running after more papers convert is safe.

Postgres rather than MySQL because this table wants three things MySQL lacks or fakes: `TEXT[]`
for categories with a GIN index, `JSONB` for the untruncated record, and built-in full-text search
— here over titles, since abstracts are not carried. See
[`src/scripts/schema.sql`](src/scripts/schema.sql) for the DDL and example queries.

```sql
SELECT primary_category, count(*) FROM papers GROUP BY 1 ORDER BY 2 DESC;
SELECT id, title FROM papers WHERE categories @> ARRAY['cs.LG'] ORDER BY date_released DESC;
```

The SQLite manifest stays regardless — crawl state needs zero-setup local durability.

---

## arXiv usage policy

This matters, so it is not buried.

arXiv's [bulk data page](https://info.arxiv.org/help/bulk_data.html) states:

> Please do not attempt to download the complete corpus programmatically.

Their sanctioned route for the whole corpus is the **S3 requester-pays bucket** (`s3://arxiv`,
~9.2 TB as of April 2025, you pay AWS egress). For anything smaller they ask that you use
`export.arxiv.org` and keep to roughly **bursts of 4 requests/second with a 1-second sleep**.

This crawler is built to respect that:

- it targets `export.arxiv.org`, not `arxiv.org`;
- a **global** token bucket caps the request rate across all workers, defaulting to a
  conservative 1 req/s;
- it sends an identifying `User-Agent` — **put a real address in `crawl.contact`**;
- it honours `Retry-After` and backs off exponentially on 429/5xx;
- `scope` filters exist so you can take a deliberate slice instead of the whole corpus.

None of that makes a full-corpus crawl appropriate. If you need everything, use the S3 bucket and
convert locally; this pipeline's `convert` stage works just as well on PDFs from there.

---

## Scale: time and disk

Measured on the smoke-test sample (8 papers, 245 pages), then extrapolated:

| | Sample | Full corpus (~2.8M) |
|---|---|---|
| PDF downloaded | 16.2 MB | ~5.4 TB *(discarded)* |
| Markdown kept | 791 KB | **~264 GB** |
| Compression | 20.9× | — |
| Tables extracted | 40 | — |
| Download @ 1 req/s | — | ~32 days |
| Download @ 4 req/s | — | ~8 days |
| Conversion, 8 processes | — | ~2 days — not the bottleneck |

Download dominates, and parallelism only helps up to the rate cap. Steady-state disk is Markdown
plus JSON; only the bounded `data/tmp/` staging directory ever holds PDFs, and it is empty when
the run finishes.

Run `python -m src.scripts.compare_size` after your own sample for figures grounded in the
categories you actually crawl — the ratio varies a lot (4.3× for a dense theory paper, 53× for a
figure-heavy one).

---

## Resuming and troubleshooting

**Interrupting is safe.** `Ctrl-C` once: in-flight work finishes, staged PDFs are cleaned up, and
rows left `in_flight` are returned to `pending` on the next start. Just run `run` again.

**Something failed.** The next `run` retries it automatically ([Retries](#retries)). `status` shows
the breakdown, and the manifest keeps a per-paper `error`. To force papers that are out of
attempts back into the queue:

```bash
.venv/bin/python -m src.main retry --stage download --max-attempts 8
```

**Outputs deleted by accident.**

```bash
.venv/bin/python -m src.main verify --fix
```

| Symptom | Cause and fix |
|---|---|
| Many `failed_download`, `HTTP 429` | Rate limited. Lower `--rps`, wait, retry. |
| `no_pdf` | Terminal, not an error: withdrawn papers and source-only submissions have no PDF. |
| `ConversionTimeout` | A pathologically long paper. Raise `convert.timeout`, then `retry --stage convert`. |
| Papers flagged `low_text` | Pre-2000 scans with no text layer. They need OCR; re-run with `--converter docling`. |
| `run` says nothing pending | The manifest is empty or complete. Run `prepare` with a wider scope. |
| Conversion retry re-downloads | Expected — the PDF was deleted. Use `--keep-pdf` when debugging. |

---

## Known limitations

- **PDFs are discarded**, so re-converting with a better backend means re-downloading.
  `pdf_sha256` and `pdf_bytes` are kept in the manifest; `--keep-pdf` overrides this for debugging.
- **Rule-only (`booktabs`) tables are missed** by `lines_strict` when a table has no vertical
  rules and no full ruling box. The only available fallback is destructive, so this is an accepted
  gap rather than a silently applied fix; `n_tables` in the metadata makes it measurable.
- **Complex merged headers collapse.** Multi-level headers and merged cells come out as a single
  cell with `<br>`-joined content. Geometric extraction cannot recover a logical span the PDF
  never recorded.
- **Two-column layouts and rotated tables** are where reading order is weakest.
- **Pre-2000 scanned papers** have no text layer at all. They convert to near-nothing and are
  flagged `low_text`.
- **Figures are dropped entirely**, by design. Captions survive as body text; the images do not.
- **Equations** become the PDF's text-layer approximation, not LaTeX. If you need real math, the
  arXiv LaTeX source is a better input than the PDF.

---

## Development

```bash
.venv/bin/python -m pytest tests/ -q
```

62 tests, no network required. The converter tests generate their fixture PDFs at run time with
PyMuPDF, so no binaries are committed.

```
src/
├── main.py                    CLI: prepare, run, status, retry, verify
├── config.py                  config.yaml -> dataclasses, with CLI overrides
├── utils/
│   ├── paths.py               ID normalisation and the yymm shard layout
│   ├── logging_setup.py       file-only logging; silences workers so the bar survives
│   ├── state.py               SQLite manifest, claim/writer machinery
│   ├── prepare_data.py        streaming JSONL ingest (shared with the notebook)
│   ├── crawler.py             rate limiter, HTTP layer, parallel orchestrator
│   ├── converter.py           PDF -> Markdown backends + the process-pool task
│   └── writer.py              atomic serialisation of the three output files
└── scripts/
    ├── compare_size.py        PDF vs Markdown size report
    ├── ingest_postgres.py     bulk load into Postgres
    └── schema.sql             Postgres DDL and example queries
```

The notebook imports `iter_records` and `parse_record` from `prepare_data`, so the EDA and the
crawler can never disagree about how a record is interpreted.
