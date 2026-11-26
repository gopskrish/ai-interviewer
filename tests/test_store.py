from datetime import UTC, datetime, timedelta

from interviewer.cleanup import main as cleanup_main
from interviewer.session.store import SessionStore


def test_store_create_get_list_newest_first(tmp_path):
    store = SessionStore(tmp_path)
    a = store.create("practice", "Asha", "senior-dotnet")
    b = store.create("interview", "Ben", "senior-dotnet", meeting_url="https://teams.microsoft.com/x", notes="n")
    a.created_at = "2026-11-25T10:00:00+00:00"
    b.created_at = "2026-11-26T10:00:00+00:00"
    store.save(a)
    store.save(b)
    assert [r.id for r in store.list()] == [b.id, a.id]
    got = store.get(b.id)
    assert got == b and got.status == "created" and got.flags == [] and got.notes == "n"
    assert len(a.id) == 12 and len(a.token) >= 24 and a.id != b.id
    assert (tmp_path / "sessions" / a.id / "session.json").exists()


def test_store_get_missing_or_unsafe_id_returns_none(tmp_path):
    store = SessionStore(tmp_path)
    assert store.get("0123456789ab") is None and store.get("../etc") is None


def test_cleanup_deletes_only_older_than_30_days(tmp_path):
    store, now = SessionStore(tmp_path), datetime(2026, 11, 26, tzinfo=UTC)
    old, recent = store.create("practice", "A", "senior-dotnet"), store.create("practice", "B", "senior-dotnet")
    old.created_at = (now - timedelta(days=31)).isoformat()
    store.save(old)
    recent.created_at = (now - timedelta(days=29)).isoformat()
    store.save(recent)
    assert store.cleanup(30, now) == [old.id] and not store.dir(old.id).exists() and store.get(recent.id)


def test_cleanup_cli(tmp_path, capsys):
    store = SessionStore(tmp_path)
    old = store.create("practice", "A", "senior-dotnet")
    old.created_at = (datetime.now(UTC) - timedelta(days=40)).isoformat()
    store.save(old)
    assert cleanup_main(["--days", "30", "--data-dir", str(tmp_path)]) == 0
    assert old.id in capsys.readouterr().out and store.get(old.id) is None
