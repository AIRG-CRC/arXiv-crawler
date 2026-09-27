"""Resuming a cut-off download with a Range request, and falling back to arxiv.org."""

from __future__ import annotations

import hashlib
import threading

import pytest

from src.utils import crawler as C
from src.utils.state import DONE, FAILED_DOWNLOAD, NO_PDF, PaperRow
from tests.test_cooldown import _Cfg, _Resp, _Session, _cooldown

FULL = b"%PDF-1.7 " + bytes(range(256)) * 40          # stands in for a real PDF
CUT = 1024                                              # where the "CDN" drops us


@pytest.fixture(autouse=True)
def _no_per_paper_backoff(monkeypatch):
    monkeypatch.setattr(C, "_sleep_for_retry", lambda response, attempt, stop: None)


class _FallbackCfg(_Cfg):
    fallback_base_url = "http://www.invalid"
    fallback_attempts = 2


def _run(tmp_path, responses, cfg=None):
    row = PaperRow(arxiv_id="2301.00001", version="v1", shard="2301")
    session = _Session(cfg or _Cfg(), responses, _cooldown(seconds=0.01))
    outcome = C.download_one(row, session, C.RateLimiter(1000.0, 1000), tmp_path,
                             threading.Event())
    return session, outcome


def _partial(start):
    return {"Content-Range": f"bytes {start}-{len(FULL) - 1}/{len(FULL)}"}


def test_a_broken_transfer_resumes_from_the_byte_it_stopped_at(tmp_path):
    session, outcome = _run(tmp_path, [
        _Resp(200, FULL[:CUT], broken=True),
        _Resp(206, FULL[CUT:], headers=_partial(CUT)),
    ])
    assert outcome.status == DONE
    assert session.calls[1][1] == f"bytes={CUT}-"
    assert outcome.path.read_bytes() == FULL
    assert outcome.size == len(FULL)
    assert outcome.sha256 == hashlib.sha256(FULL).hexdigest()   # covers both halves


def test_a_resume_can_itself_break_and_resume_again(tmp_path):
    session, outcome = _run(tmp_path, [
        _Resp(200, FULL[:CUT], broken=True),
        _Resp(206, FULL[CUT:2 * CUT], headers=_partial(CUT), broken=True),
        _Resp(206, FULL[2 * CUT:], headers=_partial(2 * CUT)),
    ])
    assert outcome.status == DONE
    assert [r for _, r in session.calls] == [None, f"bytes={CUT}-", f"bytes={2 * CUT}-"]
    assert outcome.path.read_bytes() == FULL


def test_a_server_that_ignores_range_sends_the_whole_file_again(tmp_path):
    _, outcome = _run(tmp_path, [
        _Resp(200, FULL[:CUT], broken=True),
        _Resp(200, FULL),                               # 200, not 206: start over
    ])
    assert outcome.status == DONE
    assert outcome.path.read_bytes() == FULL


def test_a_mismatched_content_range_discards_the_fragment(tmp_path):
    session, outcome = _run(tmp_path, [
        _Resp(200, FULL[:CUT], broken=True),
        _Resp(206, FULL[5:], headers=_partial(5)),      # not where we asked
        _Resp(200, FULL),
    ])
    assert outcome.status == DONE
    assert session.calls[2][1] is None                  # fresh request, no Range
    assert outcome.path.read_bytes() == FULL


def test_a_non_pdf_body_is_not_kept_for_resuming(tmp_path):
    session, _ = _run(tmp_path, [
        _Resp(200, b"<html>PDF is being generated</html>"),
        _Resp(200, FULL),
    ])
    assert session.calls[1][1] is None


def test_the_fallback_host_takes_over_from_scratch(tmp_path):
    cfg = _FallbackCfg()
    session, outcome = _run(tmp_path, [_Resp(200, FULL[:CUT], broken=True)] * 3 + [
        _Resp(200, FULL),
    ], cfg)
    assert outcome.status == DONE
    assert session.calls[3] == ("http://www.invalid/pdf/x", None)   # no spliced bytes
    assert outcome.path.read_bytes() == FULL


def test_both_hosts_failing_is_one_failure_naming_both(tmp_path):
    cfg = _FallbackCfg()
    session, outcome = _run(tmp_path, [_Resp(503)] * 5, cfg)
    assert outcome.status == FAILED_DOWNLOAD
    assert session.requests == 3 + 2                    # max_attempts + fallback_attempts
    assert "also tried www.invalid" in outcome.error
    assert not list((tmp_path / "tmp").glob("*.part"))


def test_a_404_does_not_fall_back(tmp_path):
    session, outcome = _run(tmp_path, [_Resp(404)], _FallbackCfg())
    assert outcome.status == NO_PDF
    assert session.requests == 1


def test_no_fallback_when_it_is_the_primary(tmp_path):
    class Cfg(_FallbackCfg):
        fallback_base_url = "http://export.invalid/"
    session, outcome = _run(tmp_path, [_Resp(503)] * 3, Cfg())
    assert outcome.status == FAILED_DOWNLOAD
    assert session.requests == 3
