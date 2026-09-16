"""Object naming, the dump, and the local-then-upload-then-delete ordering.

The MinIO server these are written against lives on a private network, so the SDK is
stood in for by a fake that records calls. That is the right seam anyway: what matters
here is the ordering and the naming, not that the vendor's client can talk to a socket.
"""

import json
import os

import pytest

from src.utils.migrate import count_artefacts, dump_to_bucket, iter_artefacts, prune_empty_dirs
from src.utils.objectstore import (
    ACCESS_KEY_ENV, SECRET_KEY_ENV, MinioSettings, MinioStore, ObjectStoreError,
    local_to_object, object_name, upload_and_unlink,
)


class FakeClient:
    """Records what the SDK would have been asked to do."""

    def __init__(self, *, existing=(), fail_on=()):
        self.objects: dict[str, bytes] = {name: b"" for name in existing}
        self.fail_on = set(fail_on)
        self.buckets: set[str] = set()
        self.calls: list[tuple[str, str]] = []

    def bucket_exists(self, bucket):
        return bucket in self.buckets

    def make_bucket(self, bucket):
        self.buckets.add(bucket)

    def fput_object(self, bucket, name, path):
        if name in self.fail_on:
            raise RuntimeError("upload rejected")
        self.calls.append(("put", name))
        with open(path, "rb") as fh:
            self.objects[name] = fh.read()

    def stat_object(self, bucket, name):
        if name not in self.objects:
            raise RuntimeError("not found")
        return object()


def _store(client=None, prefix="arxiv"):
    settings = MinioSettings(endpoint="host:9000", bucket="airg", prefix=prefix,
                             access_key="k", secret_key="s")
    return MinioStore(settings, client=client or FakeClient())


# --- naming ----------------------------------------------------------------------------
def test_object_names_mirror_the_local_layout():
    assert object_name("md", "2301.12345", prefix="arxiv") == "arxiv/md/2301/2301.12345.md"
    assert object_name("tables", "2301.12345", prefix="arxiv") == \
        "arxiv/tables/2301/2301.12345.tables.md"
    assert object_name("meta", "2301.12345", prefix="arxiv") == \
        "arxiv/meta/2301/2301.12345.json"


def test_old_style_ids_are_made_key_safe():
    """`hep-th/9901001` would otherwise introduce a directory nobody intended."""
    assert object_name("md", "hep-th/9901001", prefix="") == "md/9901/hep-th_9901001.md"


def test_an_empty_prefix_is_not_a_leading_slash():
    assert object_name("md", "2301.12345", prefix="") == "md/2301/2301.12345.md"


def test_local_paths_map_back_to_object_names(tmp_path):
    path = tmp_path / "md" / "2301" / "2301.12345.md"
    assert local_to_object(path, tmp_path, prefix="arxiv") == "arxiv/md/2301/2301.12345.md"


def test_files_outside_the_three_output_dirs_are_not_corpus(tmp_path):
    assert local_to_object(tmp_path / "logs" / "crawler.log", tmp_path) is None
    assert local_to_object(tmp_path / "tmp" / "2301.00001.pdf", tmp_path) is None
    assert local_to_object(tmp_path / "manifest.db", tmp_path) is None


# --- credentials -------------------------------------------------------------------------
def test_credentials_come_from_the_environment(monkeypatch):
    monkeypatch.setenv(ACCESS_KEY_ENV, "from-env")
    monkeypatch.setenv(SECRET_KEY_ENV, "also-from-env")

    class Cfg:
        endpoint, bucket, prefix, secure = "h:9000", "b", "p", False
        access_key = secret_key = None

    settings = MinioSettings.from_config(Cfg())
    assert (settings.access_key, settings.secret_key) == ("from-env", "also-from-env")
    settings.validate()


def test_missing_credentials_say_which_ones(monkeypatch):
    monkeypatch.delenv(ACCESS_KEY_ENV, raising=False)
    monkeypatch.delenv(SECRET_KEY_ENV, raising=False)
    settings = MinioSettings(endpoint="h:9000", bucket="b")
    with pytest.raises(ObjectStoreError, match=ACCESS_KEY_ENV):
        settings.validate()


# --- the upload ordering that protects the only copy --------------------------------------
def test_the_local_file_is_removed_only_after_the_upload_returns(tmp_path):
    client = FakeClient()
    store = _store(client)
    local = tmp_path / "paper.md"
    local.write_text("body")

    upload_and_unlink(store, local, "arxiv/md/2301/paper.md")

    assert client.objects["arxiv/md/2301/paper.md"] == b"body"
    assert not local.exists()


def test_a_failed_upload_leaves_the_file_on_disk(tmp_path):
    """The whole point of the ordering: a blink of the network must not lose a paper."""
    store = _store(FakeClient(fail_on={"arxiv/md/2301/paper.md"}))
    local = tmp_path / "paper.md"
    local.write_text("body")

    with pytest.raises(RuntimeError):
        upload_and_unlink(store, local, "arxiv/md/2301/paper.md")

    assert local.exists() and local.read_text() == "body"


# --- dump ----------------------------------------------------------------------------------
@pytest.fixture()
def corpus(tmp_path):
    for shard, paper in (("2301", "2301.00001"), ("2302", "2302.00002")):
        (tmp_path / "md" / shard).mkdir(parents=True, exist_ok=True)
        (tmp_path / "md" / shard / f"{paper}.md").write_text(f"# {paper}")
        (tmp_path / "meta" / shard).mkdir(parents=True, exist_ok=True)
        (tmp_path / "meta" / shard / f"{paper}.json").write_text(json.dumps({"id": paper}))
    (tmp_path / "tables" / "2301").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tables" / "2301" / "2301.00001.tables.md").write_text("| a |")
    # Not corpus: these must be left exactly where they are.
    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / "logs" / "crawler.log").write_text("noise")
    (tmp_path / "tmp").mkdir(exist_ok=True)
    (tmp_path / "tmp" / "x.pdf").write_bytes(b"%PDF-")
    return tmp_path


def test_iteration_covers_the_corpus_and_nothing_else(corpus):
    names = {p.name for p in iter_artefacts(corpus)}
    assert names == {
        "2301.00001.md", "2302.00002.md",
        "2301.00001.json", "2302.00002.json",
        "2301.00001.tables.md",
    }
    assert count_artefacts(corpus) == 5


def test_dump_uploads_everything_and_clears_local(corpus):
    client = FakeClient()
    report = dump_to_bucket(_store(client), corpus)

    assert report.uploaded == 5 and report.failed == 0
    assert "arxiv/md/2301/2301.00001.md" in client.objects
    assert "arxiv/tables/2301/2301.00001.tables.md" in client.objects
    assert not list((corpus / "md").rglob("*.md"))
    # Untouched, because they were never corpus.
    assert (corpus / "logs" / "crawler.log").exists()
    assert (corpus / "tmp" / "x.pdf").exists()


def test_keep_local_makes_it_a_copy(corpus):
    client = FakeClient()
    report = dump_to_bucket(_store(client), corpus, keep_local=True)
    assert report.uploaded == 5
    assert len(list((corpus / "md").rglob("*.md"))) == 2


def test_a_dry_run_touches_nothing(corpus):
    client = FakeClient()
    report = dump_to_bucket(_store(client), corpus, dry_run=True)
    assert report.uploaded == 5
    assert client.objects == {} and client.calls == []
    assert len(list((corpus / "md").rglob("*.md"))) == 2


def test_skip_existing_avoids_resending(corpus):
    client = FakeClient(existing={"arxiv/md/2301/2301.00001.md"})
    report = dump_to_bucket(_store(client), corpus, skip_existing=True)
    assert report.skipped == 1 and report.uploaded == 4


def test_one_bad_object_does_not_end_the_dump(corpus):
    client = FakeClient(fail_on={"arxiv/md/2301/2301.00001.md"})
    report = dump_to_bucket(_store(client), corpus)

    assert report.failed == 1 and report.uploaded == 4
    assert "2301.00001.md" in report.errors[0]
    # The failure keeps its local copy, so re-running `dump` retries exactly it.
    assert (corpus / "md" / "2301" / "2301.00001.md").exists()


def test_limit_stops_early(corpus):
    report = dump_to_bucket(_store(), corpus, limit=2)
    assert report.considered == 2


def test_empty_shard_directories_are_cleaned_up(corpus):
    dump_to_bucket(_store(), corpus)
    assert prune_empty_dirs(corpus) > 0
    assert not (corpus / "md" / "2301").exists()
