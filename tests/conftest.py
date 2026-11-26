from pathlib import Path

import pytest

from interviewer.kit import load_kit

KIT_PATH = Path(__file__).resolve().parents[1] / "kits" / "senior-dotnet.yaml"


@pytest.fixture
def kit():
    return load_kit(KIT_PATH)
