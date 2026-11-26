import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from interviewer.kit import load_kit
from tests.helpers import KITS_DIR, FakeAttendee, FakeClock, make_manager, make_settings

KIT_PATH = Path(__file__).resolve().parents[1] / "kits" / "senior-dotnet.yaml"


@pytest.fixture
def kit():
    return load_kit(KIT_PATH)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def fake_attendee():
    return FakeAttendee()


@pytest.fixture
def manager(tmp_path, clock, fake_attendee):
    return make_manager(tmp_path, clock, attendee=fake_attendee)


def _client(manager):
    from interviewer.web.app import create_app

    return TestClient(create_app(manager.settings, manager, background=False))


@pytest.fixture
def client(manager):
    with _client(manager) as test_client:
        yield test_client


@pytest.fixture
def client_with_broken_kit(tmp_path, clock, fake_attendee):
    kits_dir = tmp_path / "kits"
    kits_dir.mkdir()
    shutil.copy(KITS_DIR / "senior-dotnet.yaml", kits_dir)
    (kits_dir / "broken.yaml").write_text("role: Broken kit\ntopics: [unclosed\n", encoding="utf-8")
    manager = make_manager(tmp_path, clock, attendee=fake_attendee, settings=make_settings(tmp_path, kits_dir=kits_dir))
    with _client(manager) as test_client:
        yield test_client
