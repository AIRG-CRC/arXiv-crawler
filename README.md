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
- [Life of a paper](#life-of-a-paper)
- [Quickstart](#quickstart)
- [Output format](#output-format)
- [Command reference](#command-reference)
- [Configuration](#configuration)
- [Retries](#retries)
- [When arXiv throttles you](#when-arxiv-throttles-you)
- [Terminal output](#terminal-output)
- [Object storage](#object-storage)
- [Running on two devices](#running-on-two-devices)
- [Watching every machine](#watching-every-machine)
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
       sync ───────►│            one writer thread, results queue            │
   (bucket state)   └──────────▲──────────────────────────▲──────────────────┘
                               │                          │
  metadata JSONL ──► prepare ──┘   [pending rows]         │
                                    in this device's      │
                                    slice only            │
                                        │                 │
                          ThreadPoolExecutor(N)   ProcessPoolExecutor(M)
                          download → data/tmp/ ──► convert → md/ tables/ meta/
                            (token bucket +             └─► delete the PDF,
                             cooldown gate)                 upload if run-minio
                                     bounded queue
```

Four ideas carry the design:

**Two pools, because the work has two shapes.** Downloading is IO-bound and rate-capped, so it
runs on threads. PDF parsing is CPU-bound inside a C extension that can segfault on a malformed
file, so it runs on processes — a crash costs one worker, not the run.

**One writer.** Every manifest mutation goes through a single `ManifestWriter` thread draining a
queue. That sidesteps SQLite lock contention rather than fighting it with retries, and it means
the manifest is consistent no matter when you interrupt.

**The manifest is the source of truth — locally.** `data/manifest.db` records the state of every
paper. Resuming is just "claim the rows that aren't done yet", and `verify` cross-checks it
against what is really on disk.

**The bucket is the source of truth between devices.** Each device has its own manifest, so
neither knows what the other has finished. An object at `arxiv/md/<shard>/<id>.md` is proof that
a paper is converted, whoever converted it — so a run starts by reading the bucket back into the
manifest, and the queue is split by a hash of the arXiv id so two devices never claim the same
paper. See [Running on two devices](#running-on-two-devices).

---

## Life of a paper

What actually happens to `2301.12345`, and which module does it. This is the map to read the
code by.

| # | Step | Where |
|---|---|---|
| 1 | The snapshot record is parsed and inserted as a `pending` row, with its `yymm` shard and its partition key `crc32(id) % 256`. | [`prepare_data.py`](src/utils/prepare_data.py), [`partition.py`](src/utils/partition.py) |
| 2 | A run opens. Rows left `in_flight` by a crash go back to `pending`; the bucket is listed and anything another device already converted is marked `done`. | [`state.py`](src/utils/state.py) `reset_stale`, [`sync.py`](src/utils/sync.py) |
| 3 | The paper is **claimed** — one `UPDATE ... RETURNING`, filtered to this device's slice, so no two workers and no two devices can hold it at once. | [`state.py`](src/utils/state.py) `claim_batch` |
| 4 | A download thread waits at the cooldown gate, takes a token from the global bucket, and fetches `/pdf/2301.12345v2` into `data/tmp/`, checking the body really starts with `%PDF-`. | [`crawler.py`](src/utils/crawler.py) `download_one` |
| 5 | The staged PDF crosses into a **process** pool — bounded, so `tmp/` cannot fill if conversion falls behind. | [`crawler.py`](src/utils/crawler.py) `run_pipeline` |
| 6 | A worker converts it: body text in reading order, tables lifted out, a fallback backend if the configured one fails or times out. | [`converter.py`](src/utils/converter.py) `convert_and_write` |
| 7 | Three files are written atomically — `md`, `tables`, `meta` — and, with `run-minio`, each is uploaded and then unlinked. | [`writer.py`](src/utils/writer.py), [`objectstore.py`](src/utils/objectstore.py) |
| 8 | The PDF is deleted and the heap handed back. The result goes on a queue. | [`converter.py`](src/utils/converter.py), [`memory.py`](src/utils/memory.py) |
| 9 | One writer thread applies the result to the manifest in a batched transaction: status, sizes, page and table counts, attempt count. | [`state.py`](src/utils/state.py) `ManifestWriter` |

A failure at step 4 or 6 re-enters at step 3 — immediately if the run still has an in-run retry
for it, otherwise at the start of the next run. A throttle at step 4 is not a failure at all: it
parks every download thread until the block lifts and then re-enters at step 4 with no attempt
spent.

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

> If you installed before this note existed, re-run the install: two requirements were stuck
> together on one line in `requirements.txt`, so `minio` was silently never installed and every
> object-storage command failed with "the minio package is not installed".

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
                       [--no-retry-failed] [--retry-all] [--max-attempts N]
                       [--worker-bars] [--devices N] [--device-index I]
                       [--claim-any] [--sync | --no-sync] [--cooldown SECONDS]
```
Downloads and converts in parallel. `Ctrl-C` once to stop cleanly (in-flight work finishes, staged
PDFs are cleared, the manifest is left consistent); twice to abort.

Every run **opens with the papers an earlier run failed on**, ahead of any fresh work, so a
transient failure heals by itself — see [Retries](#retries).

| Flag | Effect |
|---|---|
| `--download-workers N` | Download threads. Raises concurrency, never the request rate. |
| `--convert-workers M` | Conversion processes. Each docling worker costs ~2 GB VRAM and ~3.7 GB RAM. |
| `--rps R` / `--burst B` | The global token bucket, shared by every download thread. |
| `--converter NAME` | Backend for this run. See [Converter backends](#converter-backends). |
| `--limit N` | Stop after N newly claimed papers. Requeues do not spend the budget. |
| `--keep-pdf` | Keep staged PDFs instead of deleting them after conversion. Debugging. |
| `--no-retry-failed` | Skip the retry-first pass and go straight to `pending`. |
| `--retry-all` | Retry every failed paper, ignoring `retry.max_attempts`. |
| `--max-attempts N` | Lifetime attempt ceiling for the retry pass. |
| `--worker-bars` | One progress line per conversion worker, under the main bar. |
| `--devices N`, `--device-index I` | This device's slice of the queue. See [Running on two devices](#running-on-two-devices). |
| `--claim-any` | Once this slice drains, take papers from outside it (bucket-checked first). |
| `--sync` / `--no-sync` | Force or skip the start-of-run bucket reconciliation. |
| `--cooldown SECONDS` | Pause length when arXiv throttles; `0` disables. See [When arXiv throttles you](#when-arxiv-throttles-you). |

Exit codes: `0` normally, `130` on `Ctrl-C`, **`75`** when the run stopped because arXiv would not
stop refusing it — a restart wrapper must treat 75 as "wait", not "try again now".

```bash
python -m src.main run-minio [same flags as run]
python -m src.main sync [--dry-run] [--devices N] [--device-index I] [--force]
python -m src.main dump [--keep-local] [--skip-existing] [--limit N] [--dry-run]
python -m src.main test-paper <id> [--converter NAME] [--minio] [--show N]
```
Object storage. `run-minio` is `run` with the output sent to MinIO instead of kept on
disk; `sync` marks papers done that another device already put in the bucket (`run-minio` does
this for you at startup); `dump` migrates a corpus that is already local; `test-paper` puts one
paper through the whole path without touching the manifest. See
[Object storage](#object-storage) and [Running on two devices](#running-on-two-devices).

```bash
python -m src.main devices      # what every machine sharing the bucket is doing
python -m src.main devices --set-devices N --auto   # reallocate the whole fleet in one write
python -m src.main devices --assign NAME=INDEX      # place one machine by hand
python -m src.main devices --auto --forget NAME    # retire a machine and re-divide
python -m src.main status       # counts by status, output size, tables extracted, last sync
python -m src.main verify       # cross-check the manifest against files on disk
python -m src.main checkpoint   # per-worker progress of the current or last run; --clear
python -m src.main retry --stage {download,convert,all} [--max-attempts N]
```
`verify --fix` re-queues papers whose outputs have vanished. It **skips papers stored remotely**,
and refuses outright if more than half of what it checked looks missing — that pattern is far more
likely to be a wrong `data_dir` than a real mass deletion. `--fix --force` overrides it.

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
| `crawl.workers` | `1` | Download threads. |
| `crawl.timeout` | `120` | Seconds per HTTP request. |
| `crawl.max_attempts` | `5` | Tries per paper per run before it is recorded as failed. |
| `crawl.chunk_size` | `65536` | Streaming read size. |
| `crawl.cooldown_statuses` | `[403, 406]` | Statuses that mean "arXiv is refusing us", not "this paper is broken". |
| `crawl.cooldown_seconds` | `3600` | First pause length. `0` disables the whole mechanism. |
| `crawl.cooldown_max_seconds` | `21600` | Ceiling on the escalating pause. |
| `crawl.cooldown_escalate` | `true` | Double the pause each round the block persists. |
| `crawl.cooldown_max_rounds` | `4` | Fruitless rounds before the run gives up (exit 75). |
| `convert.workers` | `4` | Conversion processes. `null` derives it from the core count. |
| `convert.timeout` | `1500` | Seconds per PDF; enforced with `SIGALRM` inside the worker. |
| `convert.device` | `auto` | `auto`, `cpu` or `cuda`. Pin it to catch a silent fall back to CPU. |
| `convert.num_threads` | `null` | CPU threads per worker; `null` derives `cores // workers`. |
| `convert.pdf_backend` | `pypdfium` | docling's text backend. See the note in `config.yaml`. |
| `convert.fallback_converter` | `pymupdf` | Tried when the configured backend fails. `null` disables. Set it to `null` on a machine where a silent downgrade is worse than a recorded failure. |
| `convert.max_tasks_per_child` | `100` | Retire a conversion worker after N papers, bounding heap drift. |
| `convert.memory_floor` | `0.12` | Pause dispatch below this much free RAM. Fraction, `"4GB"`, or `null`. |
| `convert.max_pages` | `300` | Longer documents are truncated, not failed. |
| `convert.table_strategy` | `lines_strict` | See [Converter backends](#converter-backends). |
| `convert.table_fallback_strategy` | `null` | Leave off — see the note in `config.yaml`. |
| `convert.min_chars_per_page` | `100` | Below this a paper is flagged `low_text`. |
| `convert.detect_pseudocode` | `true` | Keep algorithm blocks as fenced code rather than prose. |
| `convert.preserve_equations` | `true` | Keep the text layer's equation approximation inline. |
| `convert.max_table_columns` | `25` | Wider tables are treated as layout, not data. |
| `retry.on_start` | `true` | Re-attempt earlier failures at the start of every `run`. |
| `retry.max_attempts` | `4` | Total tries a paper ever gets before `run` stops picking it up. `null` = no ceiling. |
| `retry.in_run` | `true` | Retry a failed paper inside the run that failed it. |
| `retry.in_run_attempts` | `1` | Extra in-run tries before the run gives up on a paper. |
| `minio.endpoint` | `10.3.18.40:9000` | Host and port. Credentials come from the environment. |
| `minio.bucket` / `minio.prefix` | `airg` / `arxiv` | Everything lands under `<bucket>/<prefix>/`. |
| `minio.secure` | `false` | `true` for HTTPS endpoints. |
| `sync.device` | `null` | This device's name in the bucket; `null` uses the hostname. |
| `sync.devices` | `null` | How many machines share this corpus. `null` or `1` means just this one. |
| `sync.device_index` | `null` | Which slice this machine takes, 0-based. See the warning below. |
| `sync.follow_plan` | `true` | Adopt the shared allocation in the bucket when it names this device, overriding the two keys above. |
| `sync.require_meta` | `true` | A paper counts as done only if *both* its `md` and `meta` objects exist. |
| `sync.on_start` | `true` | Reconcile with the bucket at the start of every `run-minio`. |
| `sync.marker` | `true` | Publish `<prefix>/_state/sync/<device>.json` after each sync. |
| `sync.heartbeat_seconds` | `60` | How often a running crawl republishes its progress, for `devices`. `0` disables; values under 5 are raised. |

Raising `crawl.workers` increases concurrency, **never** the request rate past `rate_per_sec`.

> **`rate_per_sec` is per machine, not per corpus.** The token bucket lives in one process, so
> four machines at the default `1.0` put **4 requests/second** on export.arxiv.org — at or past
> what arXiv asks of automated clients, and a likely way to earn the 406 that the cooldown then
> waits out. Divide it by the number of devices: `crawl.rate_per_sec: 0.25` on each of four
> machines keeps the aggregate at roughly 1/s. Nothing enforces this across machines; it is
> arithmetic you have to do.

**Credentials never go in this file** — it is tracked in git. Export them:

```bash
export MINIO_ACCESS_KEY=...  MINIO_SECRET_KEY=...
```

The three per-device settings can go either in `config.yaml` or in the environment, which takes
precedence over the file; `--devices` / `--device-index` take precedence over both, so a one-off
run can always differ from the machine's usual identity.

```bash
export ARXIV_CRAWLER_DEVICE=mac-studio          # this machine's name in the bucket
export ARXIV_CRAWLER_DEVICES=2                  # how many devices share the corpus
export ARXIV_CRAWLER_DEVICE_INDEX=0             # which slice this one takes
```

> **If you set `device_index` in `config.yaml`, remember the file is tracked in git.** A pull, a
> merge or a checkout can carry one machine's index onto the other — and two devices on the same
> index crawl the same slice while nothing crawls the rest. Every run cross-checks the other
> devices' bucket markers and prints a `⚠` line when it sees that, but it warns rather than
> refuses. Read the first few lines of a run after changing these.

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

## When arXiv throttles you

Sooner or later the export host stops answering:

```
  ↻ 0708.1102 failed (HTTPError: 406 Client Error: Not Acceptable) — retrying now
  ↻ 0708.1105 failed (HTTPError: 406 Client Error: Not Acceptable) — retrying now
```

**A 406 or 403 is not a fact about the paper.** The same id fetches fine an hour later, and every
request in flight gets the same answer at the same moment — it is arXiv refusing this IP. Treated
as an ordinary error, it is the worst possible outcome: each paper burns all five of its attempts
inside a few seconds of backoff and is then recorded `failed_download` with a lifetime attempt
spent. The crawler makes its heaviest burst of requests at exactly the moment arXiv is refusing
them, and marks hundreds of perfectly good papers bad.

So a throttle stops **every** download instead:

```
  ⏸ arXiv answered HTTP 406 on 0708.1102 — pausing every download for 1h00m (cooldown 1 of 4)
  ⏸ arXiv HTTP 406 — 41m18s left ███████▌
  ▶ resuming after 1h00m
```

- The pause is **global**. One thread runs the countdown; the rest park behind a gate. A 406 that
  was already in flight when the gate closed does not start a second pause on top of the first.
- **No attempt is spent.** Waiting out a block is not a try, so the paper is fetched again
  afterwards with its retry budget intact.
- If the block is still there when the gate reopens, the wait **doubles** — 1h, 2h, 4h, 6h. Any
  civil answer resets it, and a 404 counts: it proves the host is talking to us.
- On resuming, the token bucket is **emptied** first. It had refilled to `burst` while we waited,
  and firing four requests at a server that was blocking us is how the block was earned.
- `Retry-After`, when arXiv sends one, raises the wait. It is better information than a guess.
- Conversion carries on throughout — whatever is already downloaded still gets converted.
- `Ctrl-C` interrupts the wait within a second and shuts the run down cleanly.

After `crawl.cooldown_max_rounds` rounds with nothing to show for them (~13 hours by default) the
run stops and **exits 75**. Nothing is lost: outstanding papers go back to `pending`. The non-zero
exit matters if you run under a supervisor — a wrapper that relaunches on exit 0 would walk
straight back into the block and defeat the whole mechanism.

A 200 whose body is an HTML block page is caught too. arXiv also answers 200 with a "PDF is being
generated, retry shortly" interstitial, which genuinely *is* per-paper, and only the body
distinguishes them — so the sniff is deliberately conservative and every non-PDF body is logged
with its first 200 bytes. To watch the whole mechanism without waiting an hour:

```bash
python -m src.main run --cooldown 10 --limit 5
```

---

## Terminal output

`run` prints a progress bar and nothing else. Anything worth keeping goes to
`data/logs/crawler.log`:

```
retrying 29 previously failed paper(s) first
crawl+convert:  38%|███▊      | 18,904/49,141 [2:14:07<3:34:19, 2.35paper/s, ok=18,871, fail=33, w=8]
  ✗ 0802.2167  ConversionTimeout: conversion exceeded 120s
```

The bar measures **this machine's slice**, not the whole corpus — with `--devices 2` the total is
half the corpus, so `{remaining}` is derived from a rate and a denominator that belong to the same
machine. The shared figure rides in the postfix instead:

```
crawl+convert:  21%|██▏    | 10,472/49,141 [1:02:14<2:19:41, 2.8paper/s,
                 Fail=3, Done=10,472, Workers=4/4, Pending=38,669, Corpus=20,944/98,282]
```

| Field | Means |
|---|---|
| `n/total` | Converted / reachable **in this machine's slice**. Equals the corpus on a single device. |
| `Fail`, `Done` | This run's own outcomes, since it started. |
| `Workers` | Live conversion processes out of `convert.workers`. |
| `Pending` | Papers still unclaimed **in this slice**. |
| `Corpus` | Converted / total across every machine, as of the last sync plus this run's own work. Only shown when partitioned. |

`Corpus` does not move when another machine converts something — the bucket is only re-read at the
start of a run. For a live view of the others, use [`devices`](#watching-every-machine).

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

## Running on two devices

Two machines can crawl the same corpus into the same bucket at the same time. There are two
problems to solve, and they have separate answers.

**Neither device knows what the other has finished.** Each has its own `manifest.db`. The bucket
is the shared record, so a `run-minio` starts by listing it and marking those papers `done`
locally — nothing already converted is ever crawled again.

**Both devices would otherwise claim the same papers.** `claim_batch` is atomic within one SQLite
file, and there are two files. So the queue is split by `crc32(arxiv_id) % 256`, and each device
claims only its own residue class. No coordination, no leases, no shared database.

### Changing the number of machines

The split is computed per claim — `bucket % devices == index` — so **nothing is stored that
needs migrating.** Change the count and the next run reallocates by arithmetic. Work already done
stays done; papers a machine no longer owns simply get claimed by whichever machine now owns them.

The only thing that needs agreeing is the count itself, and that lives in the bucket:

```bash
python -m src.main devices --set-devices 4 --auto
```

`--auto` assigns every machine that has reported, plus this one, to a slice in name order, and
writes `<prefix>/_state/partition.json`. **Every machine adopts it at its next run** — no
per-machine edits, and a machine you forget about cannot end up on the wrong slice:

```
allocation written to http://10.3.18.40:9000/airg/arxiv/_state/partition.json

  4 device(s), set by mac-studio at 2026-09-24T03:42:31+00:00
    slice 1 of 4  CIT
    slice 2 of 4  linux-box
    slice 3 of 4  mac-studio
    slice 4 of 4  thinkpad

every machine picks this up at its next run — no per-machine edits.
```

Each run then says which slice it took and whether that changed:

```
allocation: slice 3 of 4, from the shared plan set by mac-studio
allocation changed since the last run here: slice 1 of 2 → slice 3 of 4
  (the split is recomputed per claim; nothing needs migrating)
```

To place a machine by hand, or to add one the plan does not know about:

```bash
python -m src.main devices --assign thinkpad=3
```

**To retire a machine for good**, forget it. That deletes its marker *and* drops it from the
allocation, and the rest are reindexed contiguously:

```bash
python -m src.main devices --auto --forget mac-studio --forget old-laptop
```

Deleting the marker is the part that matters. `--auto` rebuilds the fleet from the markers, so a
retired machine whose marker survives comes straight back on the next `--auto` — remove two
machines one at a time with `--unassign` and you will watch each one undo the other. `--forget`
ends that. It refuses to forget the machine you are running it from, so run it from one that is
staying.

**`--unassign` is the other half:** it drops a machine from the allocation but keeps its marker,
for one that is only away for a while. On its own it does *not* reallocate — the count stays put
and that slice is crawled by nobody until something takes it, which `devices` then reports as a
problem. Pair it with `--auto` to re-divide, or with `--set-devices` to set the count yourself:

```bash
python -m src.main devices --set-devices 3 --unassign mac-studio    # exact, no reindexing
python -m src.main devices --auto --unassign mac-studio             # re-derive from markers
```

Lowering the count on its own cannot work: whoever held the top slice would be left pointing at a
slice that no longer exists, so `--set-devices 3` alone is refused with the two commands that fix
it.

```
  3 device(s), set by CIT at 2026-09-24T03:48:17+00:00
    slice 1 of 3  CIT
    slice 2 of 3  gx10-df62
    slice 3 of 3  linux-box
```

A machine already running keeps the split it started with, so **stop the machine you removed** —
and stop or restart the others when convenient. Nothing is lost if you don't: the three remaining
slices still cover the whole corpus, so no paper is stranded.

**One rule makes the rest predictable:** a machine that follows a plan which splits the corpus
must be named in it. A machine you dropped and then restarted refuses rather than re-crawling a
slice the others already cover — silent duplication for hours is much worse than an error you fix
in one command. `sync.follow_plan: false` is the deliberate way out.

The plan **overrides** `sync.devices` / `sync.device_index` and the environment, because a machine
with a stale local count is exactly what it exists to prevent. A `--devices` flag still wins, for a
one-off run, and `sync.follow_plan: false` pins a machine to its own settings. A machine the plan
splits away from but never names **refuses to start** rather than defaulting to slice 0 and
silently duplicating whoever owns it.

`devices` tells a rollout apart from a fault. A machine that is stopped and will adopt the new
split gets a `·` notice; one that is *crawling the wrong slice right now* gets a `⚠` and exit 1:

```
  · 'mac-studio' last ran on slice 1 of 2; it will pick up slice 3 of 3 when it next starts
  ⚠ 'linux-box' is crawling slice 2 of 2 right now, but the allocation puts it on slice 2 of 3
    — it is on the wrong slice until it restarts
```

### Setting it up

`prepare` on both devices with the **same scope** — the split is over the manifest, so different
scopes mean different slices. Then give each machine its identity, once.

Either in that machine's `config.yaml`:

```yaml
sync:
  device: linux-box     # a label for the bucket marker
  devices: 2            # how many machines share the corpus
  device_index: 1       # which slice this one takes, 0-based
```

or in its shell profile, which overrides the file:

```bash
export ARXIV_CRAWLER_DEVICE=mac-studio
export ARXIV_CRAWLER_DEVICES=2
export ARXIV_CRAWLER_DEVICE_INDEX=0
```

With `N` devices the indices are `0 … N-1`, up to 256. `--devices` / `--device-index` override
both, for a one-off run. Credentials stay in the environment either way:

```bash
export MINIO_ACCESS_KEY=...  MINIO_SECRET_KEY=...
```

Then the same command on both:

```bash
python -m src.main run-minio
```

which opens with something like:

```
storing to http://10.3.18.40:9000/airg/arxiv
  scanned 184 shard(s), skipped 41 already complete
  218,431 object(s) listed in 47.3s
  marked 1,204 paper(s) done from the bucket
device slice 1 of 2 — 1,243,905 paper(s) pending in this slice
```

### Checking it is working

```bash
python -m src.main devices
```

See [Watching every machine](#watching-every-machine). Also useful:

```bash
python -m src.main status        # "Last bucket sync: ... as device 'mac-studio'"
python -m src.main sync --dry-run
```

**The mistake that costs real work** is giving both devices the same index: they crawl the same
half of the corpus and nothing ever touches the other half, silently. It is easiest to make by
setting `device_index` in `config.yaml` and then pulling, merging or checking out that file on the
other machine — the value travels with it. If you keep the index in the config, keep an eye on
`git status` before committing. Each sync publishes
`<prefix>/_state/sync/<device>.json` recording that device's split, and every run reads the
others' markers and says so:

```
  ⚠ device 'linux-box' is also using --device-index 1: both devices will crawl the same
    slice and nothing will crawl the others
```

It warns rather than refuses — a stale marker must never stop a crawl — so the warning is worth
reading. If you see it, fix the index and run `sync` on both devices; no work is lost, the
duplicated papers are simply overwritten in place.

### Other things worth knowing

- **The slices are static**, so the faster device idles once its half is done. `--claim-any` lets
  it take papers from outside its slice at that point, checking the bucket before each download so
  it does not redo the other device's work. One HEAD per paper is affordable at the tail and
  nowhere else, which is why it is scoped there.
- **`sync` is safe to run at any time**, including during a run: papers the local run holds
  `in_flight` are left alone. It only ever marks papers done — it never re-queues anything.
- **A paper counts as done only if both its `md` and its `meta` object exist.** Uploads go md →
  tables → meta, so a device killed between the first and the last leaves an md with no meta. The
  sync reports those and leaves them for someone to finish rather than declaring victory.
- **There is deliberately no resume cursor.** A high-water mark over these key names would skip
  papers permanently: claims come back in rowid order rather than key order, old-style ids shard
  to `9901` which sorts *after* every modern `2xxx` shard, and `misc` sorts after all of them. The
  scan skips whole shards that are already complete instead, which is both sound and usually
  faster.
- **The device count can change between runs**, in one write — see
  [Changing the number of machines](#changing-the-number-of-machines). Re-slicing only means a
  given paper is crawled by a different machine next time, and the sync reconciles it either way.
  A machine already running keeps the split it started with until it restarts.
- **Divide `crawl.rate_per_sec` by the number of machines.** The rate limiter is per process, so
  N machines at the default make N requests a second against arXiv. See
  [Configuration](#configuration).
- **Check the converter on a new machine before starting a long run.** A fresh install may not be
  able to fetch its models, and every paper then silently takes the fallback — which keeps far
  fewer tables, so that machine contributes worse output to the shared bucket than the others.
  `test-paper <id> --converter docling` takes a few seconds and answers it.

---

## Watching every machine

A crawl republishes its own progress to `<prefix>/_state/sync/<device>.json` every
`sync.heartbeat_seconds`, so any machine — or a third one that is only watching — can see the rest:

```bash
python -m src.main devices
```

```
Devices sharing http://10.3.18.40:9000/airg/arxiv

device               slice       state      done  failed  papers/min   last seen
--------------------------------------------------------------------------------
mac-studio *           1/2     running    10,472       3         2.8     18s ago
linux-box              2/2    cooldown     9,918       7         2.6     60s ago
--------------------------------------------------------------------------------
mac-studio: slice 10,472/49,141 (21.3%), corpus 20,944/98,282, last synced 3h ago
linux-box: slice 9,918/49,141 (20.2%), corpus 19,836/98,282, last synced 4h ago
```

| Column | Means |
|---|---|
| `slice` | Which share of the corpus that device is taking. `all` for an unpartitioned run. |
| `state` | `running`, `finished`, `interrupted` (Ctrl-C), `throttled` (gave up on a block), or `cooldown` — currently waiting out a 406/403. |
| `done`, `failed` | That device's own outcomes in its current or last run. |
| `papers/min` | Its conversion rate, averaged over that run. |
| `last seen` | Age of its marker. Under a minute means it is alive right now. |

**Each row is that device's own last report, not a live reading.** A stale `last seen` on a
`running` row means a machine that stopped abruptly or lost the bucket — not one that finished. A
`cooldown` state with a fresh timestamp is the normal, healthy answer to "why has the other machine
stopped progressing".

`devices` also runs the partition cross-check and **exits 1** if two machines claim the same slice,
so it works as a pre-flight check:

```bash
python -m src.main devices && python -m src.main run-minio
```

It reads the bucket and nothing else — safe to run at any time, from any machine, during a crawl or
between them. `sync.heartbeat_seconds: 0` switches the publishing off if you would rather not have
one small object written per minute.

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

**Is the converter actually working on this machine?** Worth two seconds before a run measured
in days, especially on a new machine or after a driver update:

```bash
python -c "import torch, torch.nn as nn; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.cuda.is_available()); print(nn.Conv2d(3,8,3).cuda()(torch.randn(1,3,64,64,device='cuda')).shape)"
```

A cuDNN or CUDA fault shows up here immediately, with no models to download and no papers to
fetch. If it fails, `convert.device: cpu` makes docling work rather than fail — and since the
crawl is capped by `rate_per_sec` rather than by conversion, that is often fast enough to keep up.
Then confirm end to end on one paper:

```bash
python -m src.main test-paper 2102.12018 --converter docling
```

**Something failed.** The next `run` retries it automatically ([Retries](#retries)). `status` shows
the breakdown, and the manifest keeps a per-paper `error`. To force papers that are out of
attempts back into the queue:

```bash
.venv/bin/python -m src.main retry --stage download --max-attempts 8
```

**Outputs deleted by accident.** `verify` reports what is missing; `--fix` re-queues it.

```bash
.venv/bin/python -m src.main verify
.venv/bin/python -m src.main verify --fix
```

> **On a bucket-backed corpus, check what `verify` says before using `--fix`.** Papers stored in
> MinIO have no local copy — that is the point of `run-minio` — and are reported separately as
> "stored remotely", not as missing. If instead you see most of the corpus reported missing, the
> likely causes are a `data_dir` pointing somewhere unexpected, or a corpus uploaded by a version
> that did not record `remote_only`. `--fix` refuses to act on that pattern for exactly this
> reason; run `sync` first, and only use `--fix --force` if the local files really are gone.

| Symptom | Cause and fix |
|---|---|
| `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` | docling reached the GPU but PyTorch cannot load a cuDNN sublibrary — a broken or mismatched CUDA install, not a problem with the paper. Reproduce it in two seconds with the conv2d check below; then either fix the install or set `convert.device: cpu` on that machine. |
| Everything `converted by fallback` | The configured converter is broken on this machine, not defeated by the papers. The run says so once after 10 consecutive fallbacks and again in the summary. For docling, a `RepositoryNotFoundError: 401` means an HF token in the environment is being sent and rejected — the model repo is public, so `unset HF_TOKEN HUGGING_FACE_HUB_TOKEN` and verify with `test-paper <id> --converter docling` before restarting. |
| `⚠ device 'X' last ran with --devices N` | Nobody has set a shared allocation, so each machine is using its own count and they disagree. `devices --set-devices N --auto` fixes it in one write. |
| `· 'X' last ran on slice … it will pick up …` | Not a problem — a stopped machine that will adopt the new allocation when it restarts. Only the `⚠` lines need action. |
| `the shared allocation … does not name this device` | Either this machine should be in the allocation (`devices --assign <name>=<n>`, or `devices --auto`), or it was deliberately removed and should be stopped. It refuses to fall back to its own config because that would duplicate whoever now owns that share. |
| `… is assigned a slice that does not exist` | You lowered the count without saying which machine is leaving. `devices --auto --forget <name>` retires one and reindexes the rest. |
| `--auto` keeps bringing back a machine you removed | Its marker is still in the bucket and `--auto` rebuilds the fleet from the markers. `--forget <name>` deletes the marker; `--unassign` alone does not. |
| `HTTP 406` or `403`, many papers at once | arXiv is refusing this IP, not rejecting the papers. Handled automatically — see [When arXiv throttles you](#when-arxiv-throttles-you). If it keeps happening, lower `--rps`. |
| The run exited 75 | It waited out the full cooldown ladder and arXiv never relented. Wait a few hours. Do not auto-restart on 75. |
| Many `failed_download`, `HTTP 429` | Rate limited. Lower `--rps`, wait, retry. |
| `no_pdf` | Terminal, not an error: withdrawn papers and source-only submissions have no PDF. |
| `ConversionTimeout` | A pathologically long paper. Raise `convert.timeout`, then `retry --stage convert`. |
| Papers flagged `low_text` | Pre-2000 scans with no text layer. They need OCR; re-run with `--converter docling`. |
| `run` says nothing pending | The manifest is empty or complete — or this device's slice is. Check `--device-index`, or run `prepare` with a wider scope. |
| `verify` says everything is missing | The corpus is in the bucket. See the warning above. |
| Both devices are crawling the same papers | They share a `--device-index`. See [Running on two devices](#running-on-two-devices). |
| `bucket sync skipped (...)` | The bucket was unreachable. The run continues from the local manifest; fix the endpoint or credentials and it reconciles next time. |
| Conversion retry re-downloads | Expected — the PDF was deleted. Use `--keep-pdf` when debugging. |
| First start after an update is slow | A one-off migration filling the partition key for existing rows. ~8s for 2.7M papers; it is logged. |

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
- **The device split is static.** Each device gets a fixed hash slice, so the faster one idles once
  its slice is done unless you pass `--claim-any`; and a device that dies mid-run leaves its slice
  untouched until it runs again — there is no lease for another device to pick up. Genuine
  distributed claiming would need a shared manifest (Postgres, say) rather than one SQLite file per
  device. At two devices that machinery buys nothing the hash split and the bucket sync do not
  already give.
- **The bucket scan is a full listing** of every shard that still has unfinished papers in it. That
  is a few thousand LIST calls on a large corpus — tens of seconds on a LAN, and correct, which a
  stored cursor would not be.

---

## Development

```bash
.venv/bin/python -m pytest tests/ -q
```

295 tests, no network required. The converter suite is skipped unless `pymupdf` is installed and
one memory test is Linux-only, so a clean macOS checkout reports `249 passed, 2 skipped`. The
converter tests generate their fixture PDFs at run time, so no binaries are committed; the
cooldown tests drive a stubbed HTTP response and the sync tests a fake MinIO client, so nothing
reaches the network.

```
src/
├── main.py                    CLI: prepare, run, run-minio, sync, status, retry, verify
├── config.py                  config.yaml -> dataclasses, with CLI overrides
├── utils/
│   ├── paths.py               ID normalisation and the yymm shard layout
│   ├── partition.py           which papers belong to this device
│   ├── logging_setup.py       file-only logging; silences workers so the bar survives
│   ├── state.py               SQLite manifest, claim/writer machinery, migrations
│   ├── prepare_data.py        streaming JSONL ingest (shared with the notebook)
│   ├── crawler.py             rate limiter, HTTP layer, parallel orchestrator
│   ├── cooldown.py            the global pause for when arXiv stops answering
│   ├── converter.py           PDF -> Markdown backends + the process-pool task
│   ├── writer.py              atomic serialisation of the three output files
│   ├── objectstore.py         MinIO: object names, uploads, listings
│   ├── sync.py                bucket -> manifest reconciliation, device markers
│   ├── migrate.py             bulk upload of a corpus already on disk
│   ├── checkpoint.py          per-worker progress files
│   └── memory.py              the free-RAM guard and the per-paper heap release
└── scripts/
    ├── compare_size.py        PDF vs Markdown size report
    ├── ingest_postgres.py     bulk load into Postgres
    └── schema.sql             Postgres DDL and example queries
```

The notebook imports `iter_records` and `parse_record` from `prepare_data`, so the EDA and the
crawler can never disagree about how a record is interpreted.
