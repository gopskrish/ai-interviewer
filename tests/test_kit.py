import shutil
from pathlib import Path

import pytest
import yaml

from interviewer.kit import KitError, list_kits, load_kit

KIT = Path(__file__).resolve().parents[1] / "kits" / "senior-dotnet.yaml"


def write(tmp_path, mutate=None, name="kit.yaml", **overrides):
    data = yaml.safe_load(KIT.read_text(encoding="utf-8"))
    data.update(overrides)
    if mutate:
        mutate(data)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_senior_dotnet_kit_loads():
    kit = load_kit(KIT)
    assert kit.role == "Senior .NET Developer" and kit.duration_minutes == 30
    assert kit.interview_topics == ["csharp_runtime", "aspnet_api", "system_design"]
    assert len(kit.topics) == 7
    assert kit.stage_minutes.model_dump() == {
        "intro": 1, "warmup": 2, "experience": 5, "technical": 18, "candidate_questions": 3, "close": 1,
    }
    assert "Is that OK with you?" in kit.intro_script
    assert kit.topic("aspnet_api").id == "aspnet_api"


def test_kit_requires_ai_disclosure(tmp_path):
    with pytest.raises(KitError) as e:
        load_kit(write(tmp_path, intro_script="Hello, this is recorded. Is that OK with you?"))
    assert any("intro_script" in p for p in e.value.problems)


def test_kit_rubric_needs_levels_1_to_4(tmp_path):
    def mutate(d):
        d["topics"][0]["rubric"] = {1: "a", 2: "b", 3: "c"}

    with pytest.raises(KitError) as e:
        load_kit(write(tmp_path, mutate))
    assert any("rubric" in p for p in e.value.problems)


def test_kit_unknown_interview_topic(tmp_path):
    with pytest.raises(KitError) as e:
        load_kit(write(tmp_path, interview_topics=["nope"]))
    assert any("nope" in p for p in e.value.problems)


def test_kit_stage_minutes_exceed_duration(tmp_path):
    def mutate(d):
        d["stage_minutes"]["technical"] = 40

    with pytest.raises(KitError) as e:
        load_kit(write(tmp_path, mutate))
    assert any("stage_minutes" in p for p in e.value.problems)


def test_kit_yaml_syntax_error_is_kit_error(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("role: [\n", encoding="utf-8")
    with pytest.raises(KitError) as e:
        load_kit(path)
    assert "line" in e.value.problems[0]


def test_list_kits_reports_errors_per_file(tmp_path):
    shutil.copy(KIT, tmp_path / "good.yaml")
    (tmp_path / "broken.yaml").write_text("role: [\n", encoding="utf-8")
    kits, errors = list_kits(tmp_path)
    assert list(kits) == ["good"]
    assert list(errors) == ["broken"] and errors["broken"]
