import xml.etree.ElementTree as ET

import pytest

from src.config import Config
from src.utils.harvest import (
    WATERMARK_KEY, ApiClient, HarvestError, OaiClient, api_entry_to_snapshot_record, harvest,
    make_client, resolve_since, split_authors, to_snapshot_record,
)
from src.utils.prepare_data import parse_record
from src.utils.state import (
    DONE, FAILED_DOWNLOAD, IN_FLIGHT, OUTDATED, PENDING, Manifest, PaperRow,
)

OAI = "http://www.openarchives.org/OAI/2.0/"


def _record(arxiv_id, versions, *, cats="cs.LG", authors="Ada Lovelace and Alan M. Turing",
            datestamp="2026-11-05", deleted=False):
    if deleted:
        return (f'<record><header status="deleted"><identifier>oai:arXiv.org:{arxiv_id}'
                f'</identifier><datestamp>{datestamp}</datestamp></header></record>')
    vers = "".join(
        f'<version version="{v}"><date>{d}</date><size>1kb</size></version>'
        for v, d in versions)
    return f"""
    <record><header><identifier>oai:arXiv.org:{arxiv_id}</identifier>
      <datestamp>{datestamp}</datestamp></header>
      <metadata><arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/">
        <id>{arxiv_id}</id>{vers}
        <title>A   paper
          title</title>
        <authors>{authors}</authors>
        <categories>{cats}</categories>
        <doi>10.1/x</doi>
      </arXivRaw></metadata></record>"""


def _page(records, token=None):
    tok = f'<resumptionToken cursor="0">{token}</resumptionToken>' if token else ""
    return (f'<OAI-PMH xmlns="{OAI}"><ListRecords>{"".join(records)}{tok}'
            f'</ListRecords></OAI-PMH>').encode()


class _Resp:
    def __init__(self, content, status=200, headers=None):
        self.content, self.status_code, self.headers = content, status, headers or {}


class _Session:
    """Serves canned responses in order and remembers what was asked."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.urls = []
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params))
        self.urls.append(url)
        return self.responses.pop(0)


def _one(record_xml):
    root = ET.fromstring(_page([record_xml]))
    return root.find(f"{{{OAI}}}ListRecords/{{{OAI}}}record")


def _client(cfg, responses):
    session = _Session(responses)
    return OaiClient(cfg.update, "test@example.org", session=session,
                     sleep=lambda s: None), session


@pytest.fixture()
def cfg(tmp_path):
    c = Config()
    c.paths.data_dir = tmp_path
    c.paths.__post_init__()
    return c


@pytest.fixture()
def manifest(tmp_path):
    with Manifest(tmp_path / "m.db") as m:
        yield m


V1 = ("v1", "Mon, 2 Nov 2026 10:00:00 GMT")
V2 = ("v2", "Thu, 5 Nov 2026 10:00:00 GMT")


def test_split_authors_handles_and_initials_and_affiliations():
    assert split_authors("Ada Lovelace, Alan M. Turing (Bletchley), and Grace Hopper Jr") == [
        ["Lovelace", "Ada", ""], ["Turing", "Alan M.", ""], ["Hopper", "Grace", "Jr"]]


def test_record_converts_to_the_snapshot_shape():
    rec = to_snapshot_record(_one(_record("2611.00001", [V1, V2])))
    parsed = parse_record(rec)
    assert parsed["id"] == "2611.00001"
    assert parsed["version"] == "v2"
    assert parsed["title"] == "A paper title"
    assert parsed["authors"] == ["Lovelace, Ada", "Turing, Alan M."]
    assert parsed["date_released"] == "2026-11-02"
    assert parsed["date_updated"] == "2026-11-05"
    assert parsed["doi"] == "10.1/x"


def test_deleted_records_are_skipped():
    assert to_snapshot_record(_one(_record("2611.1", [], deleted=True))) is None


def test_harvest_follows_the_resumption_token_and_stores_the_watermark(cfg, manifest):
    client, session = _client(cfg, [
        _Resp(_page([_record("2611.00001", [V1])], token="abc")),
        _Resp(_page([_record("2611.00002", [V1])])),
    ])
    counts = harvest(cfg, manifest, since="2026-11-01", until="2026-11-08", client=client)
    assert counts["inserted"] == 2
    assert session.calls[0]["from"] == "2026-11-01"
    assert session.calls[1] == {"verb": "ListRecords", "resumptionToken": "abc"}
    assert manifest.stats()[PENDING] == 2
    assert manifest.get_kv(WATERMARK_KEY) == "2026-11-08"


def test_scope_filters_harvested_records(cfg, manifest):
    cfg.scope.categories = ["cs.CL"]
    client, _ = _client(cfg, [_Resp(_page([
        _record("2611.00001", [V1], cats="cs.CL"),
        _record("2611.00002", [V1], cats="math.AG"),
    ]))])
    counts = harvest(cfg, manifest, since="2026-11-01", client=client)
    assert (counts["seen"], counts["matched"], counts["inserted"]) == (2, 1, 1)


def test_new_versions_requeue_done_papers_and_spare_in_flight(cfg, manifest):
    manifest.add_papers([
        PaperRow(arxiv_id=i, version="v1", shard="2611") for i in
        ("2611.00001", "2611.00002", "2611.00003", "2611.00004")])
    manifest.conn.executemany("UPDATE papers SET status = ?, attempts = 2 WHERE arxiv_id = ?", [
        (DONE, "2611.00001"), (IN_FLIGHT, "2611.00002"), (FAILED_DOWNLOAD, "2611.00004")])
    manifest.conn.commit()
    client, _ = _client(cfg, [_Resp(_page([
        _record("2611.00001", [V1, V2]),     # done -> outdated
        _record("2611.00002", [V1, V2]),     # in flight -> untouched
        _record("2611.00003", [V1]),         # same version -> untouched
        _record("2611.00004", [V1, V2]),     # failed: version bumped, status kept
    ]))])
    counts = harvest(cfg, manifest, since="2026-11-01", client=client)
    assert (counts["version_bumped"], counts["requeued"]) == (2, 1)
    got = {r["arxiv_id"]: (r["version"], r["status"], r["attempts"]) for r in
           manifest.conn.execute("SELECT arxiv_id, version, status, attempts FROM papers")}
    assert got["2611.00001"] == ("v2", OUTDATED, 0)
    assert got["2611.00002"] == ("v1", IN_FLIGHT, 2)
    assert got["2611.00003"] == ("v1", PENDING, 0)
    assert got["2611.00004"] == ("v2", FAILED_DOWNLOAD, 2)
    # Outdated papers are handed out by an ordinary claim.
    assert {r.arxiv_id for r in manifest.claim_batch(10)} == {"2611.00001", "2611.00003"}


def test_recrawl_can_be_switched_off(cfg, manifest):
    cfg.update.recrawl_new_versions = False
    manifest.add_papers([PaperRow(arxiv_id="2611.00001", version="v1", shard="2611")])
    manifest.conn.execute("UPDATE papers SET status = ?", (DONE,))
    manifest.conn.commit()
    client, _ = _client(cfg, [_Resp(_page([_record("2611.00001", [V1, V2])]))])
    harvest(cfg, manifest, since="2026-11-01", client=client)
    row = manifest.conn.execute("SELECT version, status FROM papers").fetchone()
    assert tuple(row) == ("v2", DONE)


def test_dry_run_writes_nothing(cfg, manifest):
    client, _ = _client(cfg, [_Resp(_page([_record("2611.00001", [V1])]))])
    counts = harvest(cfg, manifest, since="2026-11-01", dry_run=True, client=client)
    assert counts["inserted"] == 1
    assert manifest.stats()["total"] == 0
    assert manifest.get_kv(WATERMARK_KEY) is None


def test_no_records_match_is_an_empty_harvest(cfg, manifest):
    body = f'<OAI-PMH xmlns="{OAI}"><error code="noRecordsMatch">none</error></OAI-PMH>'
    client, _ = _client(cfg, [_Resp(body.encode())])
    assert harvest(cfg, manifest, since="2026-11-01", client=client)["seen"] == 0


def test_503_is_retried_after_the_server_says(cfg, manifest):
    waits = []
    session = _Session([_Resp(b"", 503, {"Retry-After": "7"}),
                        _Resp(_page([_record("2611.00001", [V1])]))])
    client = OaiClient(cfg.update, "x", session=session, sleep=waits.append)
    assert harvest(cfg, manifest, since="2026-11-01", client=client)["inserted"] == 1
    assert waits[0] == 7


def test_since_falls_back_to_watermark_then_snapshot(manifest):
    with pytest.raises(HarvestError):
        resolve_since(manifest, 2)
    manifest.add_papers([PaperRow(arxiv_id="2611.00001", version="v1", shard="2611",
                                  date_updated="2026-10-31")])
    assert resolve_since(manifest, 2) == "2026-10-29"
    manifest.set_kv(WATERMARK_KEY, "2026-11-08")
    assert resolve_since(manifest, 2) == "2026-11-06"
    assert resolve_since(manifest, 2, "2026-01-01") == "2026-01-01"


class _Clock:
    """Each request appears to take `step` seconds."""

    def __init__(self, step):
        self.t, self.step = 0.0, step

    def __call__(self):
        self.t += self.step / 2
        return self.t


def test_a_slow_503_is_a_timeout_and_backs_off_for_minutes(cfg, manifest):
    waits = []
    session = _Session([_Resp(b"", 503, {"Retry-After": "1"}),
                        _Resp(b"", 503, {"Retry-After": "1"}),
                        _Resp(_page([_record("2611.00001", [V1])]))])
    client = OaiClient(cfg.update, "x", session=session, sleep=waits.append,
                       clock=_Clock(185))
    assert harvest(cfg, manifest, since="2026-11-01", client=client)["inserted"] == 1
    assert waits == [60, 120]                    # not the server's meaningless 1s


def test_backoff_is_capped(cfg):
    cfg.update.max_backoff = 100
    client, _ = _client(cfg, [])
    assert [client._backoff(n) for n in (1, 2, 3)] == [60, 100, 100]


def test_falls_back_to_the_second_endpoint_before_the_first_page(cfg, manifest):
    cfg.update.fallback_oai_url = "https://mirror.example/oai"
    cfg.update.max_attempts = 2
    client, session = _client(cfg, [
        _Resp(b"", 503), _Resp(b"", 503),                     # primary: exhausted
        _Resp(_page([_record("2611.00001", [V1])])),           # fallback: fine
    ])
    assert harvest(cfg, manifest, since="2026-11-01", client=client)["inserted"] == 1
    assert session.urls == [cfg.update.oai_url] * 2 + [cfg.update.fallback_oai_url]


def test_no_fallback_once_pages_have_been_read(cfg, manifest):
    """A resumption token is only valid on the server that issued it."""
    cfg.update.fallback_oai_url = "https://mirror.example/oai"
    cfg.update.max_attempts = 1
    client, session = _client(cfg, [
        _Resp(_page([_record("2611.00001", [V1])], token="abc")),
        _Resp(b"", 503),
    ])
    with pytest.raises(HarvestError):
        harvest(cfg, manifest, since="2026-11-01", client=client)
    assert set(session.urls) == {cfg.update.oai_url}
    assert manifest.get_kv(WATERMARK_KEY) is None         # nothing claimed as harvested


def test_fallback_can_be_disabled(cfg, manifest):
    cfg.update.fallback_oai_url = None
    cfg.update.max_attempts = 1
    client, session = _client(cfg, [_Resp(b"", 503)])
    with pytest.raises(HarvestError):
        harvest(cfg, manifest, since="2026-11-01", client=client)
    assert len(session.urls) == 1


def test_429_backs_off_instead_of_aborting(cfg, manifest):
    waits = []
    session = _Session([_Resp(b"", 429, {"Retry-After": "5"}),
                        _Resp(_page([_record("2611.00001", [V1])]))])
    client = OaiClient(cfg.update, "x", session=session, sleep=waits.append)
    assert harvest(cfg, manifest, since="2026-11-01", client=client)["inserted"] == 1
    assert waits == [60]                          # at least a full backoff, not 5s


# --- the search API -----------------------------------------------------------------------
ATOM = "http://www.w3.org/2005/Atom"


def _entry(arxiv_id, version="v1", *, published="2026-11-02T10:00:00Z",
           updated="2026-11-05T10:00:00Z", primary="cs.LG", others=("cs.AI",)):
    cats = "".join(f'<category term="{c}"/>' for c in (primary, *others))
    return f"""<entry>
      <id>http://arxiv.org/abs/{arxiv_id}{version}</id>
      <title>A   paper
        title</title>
      <updated>{updated}</updated><published>{published}</published>
      {cats}<arxiv:primary_category term="{primary}"/>
      <arxiv:doi>10.1/x</arxiv:doi>
      <author><name>Ada Lovelace</name></author><author><name> Kolyang</name></author>
    </entry>"""


def _feed(entries, total=None):
    total = len(entries) if total is None else total
    return (f'<feed xmlns="{ATOM}" xmlns:arxiv="http://arxiv.org/schemas/atom" '
            f'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
            f'<opensearch:totalResults>{total}</opensearch:totalResults>'
            f'{"".join(entries)}</feed>').encode()


def _api(cfg, responses, categories=None):
    session = _Session(responses)
    return ApiClient(cfg.update, "x", categories=categories, session=session,
                     sleep=lambda s: None), session


def test_api_entry_converts_to_the_snapshot_shape():
    root = ET.fromstring(_feed([_entry("2603.04399", "v3", published="2026-03-04T08:00:00Z")]))
    rec = api_entry_to_snapshot_record(root.find(f"{{{ATOM}}}entry"))
    parsed = parse_record(rec)
    assert parsed["id"] == "2603.04399"
    assert parsed["version"] == "v3"
    assert parsed["date_released"] == "2026-03-04"
    assert parsed["date_updated"] == "2026-11-05"
    assert parsed["primary_category"] == "cs.LG"
    assert parsed["categories"] == ["cs.LG", "cs.AI"]
    assert parsed["authors"] == ["Lovelace, Ada", "Kolyang"]
    assert parsed["title"] == "A paper title"
    assert parsed["doi"] == "10.1/x"


def test_old_style_ids_keep_their_archive():
    root = ET.fromstring(_feed([_entry("hep-th/9901001", "v2")]))
    rec = api_entry_to_snapshot_record(root.find(f"{{{ATOM}}}entry"))
    assert rec["id"] == "hep-th/9901001"


def test_api_pages_newest_first_and_stops_before_since(cfg, manifest):
    cfg.update.api_page_size = 2
    client, session = _api(cfg, [
        _Resp(_feed([_entry("2611.00003", updated="2026-11-07T09:00:00Z"),
                     _entry("2611.00002", updated="2026-11-06T09:00:00Z")], total=50)),
        _Resp(_feed([_entry("2611.00001", updated="2026-11-05T09:00:00Z"),
                     _entry("2610.00009", updated="2026-10-20T09:00:00Z")], total=50)),
    ], categories=["cs.LG", "cs.CL"])
    counts = harvest(cfg, manifest, since="2026-11-01", until="2026-11-08", client=client)
    assert counts["inserted"] == 3                       # the October one is out of range
    assert len(session.calls) == 2                       # and nothing past it was fetched
    assert session.calls[0]["search_query"] == "cat:cs.LG OR cat:cs.CL"
    assert session.calls[0]["sortBy"] == "lastUpdatedDate"
    assert session.calls[1]["start"] == "2"
    assert manifest.get_kv(WATERMARK_KEY) == "2026-11-08"


def test_api_finds_new_versions_of_old_papers(cfg, manifest):
    manifest.add_papers([PaperRow(arxiv_id="2603.04399", version="v1", shard="2603")])
    manifest.conn.execute("UPDATE papers SET status = ?", (DONE,))
    manifest.conn.commit()
    client, _ = _api(cfg, [_Resp(_feed([_entry("2603.04399", "v2",
                                               published="2026-03-04T08:00:00Z")]))])
    counts = harvest(cfg, manifest, since="2026-11-01", client=client)
    assert counts["requeued"] == 1
    row = manifest.conn.execute("SELECT version, status FROM papers").fetchone()
    assert tuple(row) == ("v2", OUTDATED)


def test_an_empty_page_mid_listing_is_retried(cfg, manifest):
    cfg.update.api_page_size = 1
    client, session = _api(cfg, [
        _Resp(_feed([_entry("2611.00002", updated="2026-11-06T09:00:00Z")], total=2)),
        _Resp(_feed([], total=2)),                                    # the API's hiccup
        _Resp(_feed([_entry("2611.00001", updated="2026-11-05T09:00:00Z")], total=2)),
    ])
    assert harvest(cfg, manifest, since="2026-11-01", client=client)["inserted"] == 2
    assert [c["start"] for c in session.calls] == ["0", "1", "1"]


def test_a_gap_beyond_the_30000_ceiling_keeps_the_watermark(cfg, manifest, monkeypatch):
    monkeypatch.setattr(ApiClient, "MAX_RESULTS", 2)
    cfg.update.api_page_size = 2
    manifest.set_kv(WATERMARK_KEY, "2026-09-01")
    client, session = _api(cfg, [
        _Resp(_feed([_entry("2611.00002", updated="2026-11-06T09:00:00Z"),
                     _entry("2611.00001", updated="2026-11-05T09:00:00Z")], total=99)),
    ])
    counts = harvest(cfg, manifest, since="2026-09-01", client=client)
    assert counts["inserted"] == 2                      # the newest papers still land
    assert counts["complete"] is False
    assert manifest.get_kv(WATERMARK_KEY) == "2026-09-01"


def test_the_source_is_chosen_by_config(cfg):
    assert isinstance(make_client(cfg), ApiClient)
    cfg.scope.categories = ["cs.LG"]
    assert make_client(cfg).query == "cat:cs.LG"
    cfg.update.source = "oai"
    assert isinstance(make_client(cfg), OaiClient)
    cfg.update.source = "rss"
    with pytest.raises(HarvestError):
        make_client(cfg)
