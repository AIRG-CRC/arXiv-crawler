"""Where log records go, and — mostly — where they do not.

A run is a progress bar, not a transcript. Two things fought that:

  * `docling` logs four INFO lines per paper ("detected formats", "Going to convert
    document batch", "Processing document", "Finished converting"). At corpus scale that
    is millions of lines: one observed `crawler.log` reached 67 MB, and the terminal
    scrolled far too fast to read the bar;
  * conversion runs in a *process* pool. On Linux those processes are forked, so they
    inherit the parent's root handlers — every worker writes the same flood to the same
    stderr and the same file.

So the console gets nothing by default. The bar is the interface, failures are printed
through `tqdm.write` by the orchestrator so they cannot corrupt it, and everything else
is kept in ``data/logs/crawler.log`` for afterwards. ``-v`` puts the full stream back on
stderr when you actually want to watch it.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any

# Third-party loggers that are chatty at INFO and say nothing this pipeline acts on.
# Their WARNING and ERROR records still reach the log file.
NOISY_LOGGERS = (
    "docling",
    "docling_core",
    "docling_ibm_models",
    "docling_parse",
    # Not namespaced under `docling` -- docling_ibm_models' table matcher registers a
    # bare top-level logger and emits one WARNING per recovered table cell.
    "MatchingPostProcessor",
    "urllib3",
    "requests",
    "filelock",
    "fsspec",
    "huggingface_hub",
    "matplotlib",
    "PIL",
    "pypdfium2",
    "torch",
    "transformers",
)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(logs_dir: Path, *, verbose: bool = False, quiet: bool = True) -> None:
    """Install the process-wide logging configuration.

    `quiet` (the default for `run`) means: file only, no console handler at all.
    `verbose` overrides it and restores the old stderr stream, DEBUG level included.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.FileHandler(logs_dir / "crawler.log", encoding="utf-8")
    ]
    if verbose or not quiet:
        handlers.append(logging.StreamHandler(sys.stderr))

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,
    )
    if not verbose:
        for name in NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)


def quiet_worker_logging() -> None:
    """`ProcessPoolExecutor` initializer: shut a conversion worker up.

    Runs once per worker process. Under `fork` it undoes the inherited console handler;
    under `spawn` it configures a process that has no handlers yet. Either way the worker
    reports outcomes through the `TaskResult` it returns, never through the terminal.
    """
    # tqdm honours this from the environment, which is how docling's internal page bars
    # are suppressed without reaching into its API. Set inside the child only, so the
    # orchestrator's own bar in the parent process is untouched.
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    root = logging.getLogger()
    for handler in list(root.handlers):
        # Keep an inherited FileHandler -- a genuine ERROR from a worker is worth having
        # on disk -- but drop anything aimed at the terminal, bar included.
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
    root.setLevel(logging.ERROR)

    # Raising the root level is not enough on its own. A logger only defers to root when
    # its own level is NOTSET, and docling's table matcher configures itself -- so
    # `MatchingPostProcessor` kept emitting a WARNING per recovered table cell straight
    # past a root set to ERROR: 2,546 lines in four minutes, into a log already at 67 MB.
    # `logging.disable` is checked before any per-logger level, so it is the only switch
    # that holds regardless of what a dependency does to its own loggers.
    logging.disable(logging.WARNING)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")


def console(message: str, *args: Any) -> None:
    """Write one line to the terminal without breaking an active progress bar."""
    text = message % args if args else message
    try:
        from tqdm import tqdm

        tqdm.write(text)
    except Exception:                       # tqdm missing or stream already closed
        print(text, file=sys.stderr)
