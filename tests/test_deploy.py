"""The compose stack, setup script and packaging, checked without running Docker."""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.helpers import REPO

COMPOSE = REPO / "deploy" / "docker-compose.yml"
SETUP = REPO / "scripts" / "setup.sh"
SERVICES = {"attendee-app", "attendee-worker", "attendee-scheduler", "postgres", "redis", "minio", "minio-init",
            "ollama", "interviewer"}


def services():
    return yaml.safe_load(COMPOSE.read_text())["services"]


def env_of(service: dict) -> dict:
    env = service.get("environment", {})
    if isinstance(env, list):
        env = dict(item.split("=", 1) for item in env)
    return env


def test_setup_script_syntax():
    assert subprocess.run(["bash", "-n", str(SETUP)]).returncode == 0
    assert subprocess.run(["bash", "-n", str(REPO / "scripts" / "package.sh")]).returncode == 0


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_compose_config_valid(tmp_path):
    made = []
    for path in (REPO / ".env", REPO / "deploy" / "attendee.env"):
        if not path.exists():
            path.write_text("")
            made.append(path)
    try:
        result = subprocess.run(["docker", "compose", "-f", str(COMPOSE), "config", "-q"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    finally:
        for path in made:
            path.unlink()


def test_compose_pins_attendee_and_localhost_ports():
    text = COMPOSE.read_text()
    assert "SSL_CERT_FILE=/certs/ca-bundle.pem" in text and "ALLOWED_HOSTS=localhost,attendee-app" in text
    assert "127.0.0.1:8443:8443" in text and "127.0.0.1:8000:8000" in text
    assert "e7a0b829867b6f6413a0118878a81ddb327b7681" in SETUP.read_text()


def test_compose_services_and_wiring():
    svc = services()
    assert set(svc) == SERVICES
    for name in ("attendee-app", "attendee-worker", "attendee-scheduler"):
        attendee = svc[name]
        assert attendee["build"] == "../vendor/attendee" and attendee["env_file"] == ["attendee.env"]
        env = env_of(attendee)
        assert env["DJANGO_SETTINGS_MODULE"] == "attendee.settings.development"
        assert env["POSTGRES_HOST"] == "postgres" and env["REDIS_URL"] == "redis://redis:6379/5"
    worker = svc["attendee-worker"]
    assert env_of(worker)["ENABLE_CHROME_SANDBOX"] == "false" and "../certs:/certs:ro" in worker["volumes"]
    assert svc["attendee-app"]["command"] == "python manage.py runserver 0.0.0.0:8000"
    assert worker["command"] == "celery -A attendee worker -l INFO"
    assert svc["attendee-scheduler"]["command"] == "python manage.py run_scheduler"
    assert svc["postgres"]["image"] == "postgres:15.3-alpine" and svc["redis"]["image"] == "redis:7-alpine"
    pg = env_of(svc["postgres"])
    assert (pg["POSTGRES_DB"], pg["POSTGRES_USER"], pg["POSTGRES_PASSWORD"]) == (
        "attendee_development", "attendee_development_user", "attendee_development_user")
    assert "attendee-recordings" in str(svc["minio-init"]) and svc["minio"]["command"] == "server /data"
    assert "ollama:/root/.ollama" in svc["ollama"]["volumes"]
    interviewer = svc["interviewer"]
    assert interviewer["build"] == {"context": "..", "dockerfile": "deploy/interviewer.Dockerfile"}
    assert set(interviewer["volumes"]) == {"../data:/data", "../models:/models", "../certs:/certs:ro",
                                           "../kits:/app/kits:ro"}
    assert interviewer["ports"] == ["127.0.0.1:8443:8443"] and interviewer["env_file"] == ["../.env"]
    assert interviewer["command"] == ("uvicorn interviewer.web.app:create_app --factory --host 0.0.0.0 --port 8443 "
                                      "--ssl-keyfile /certs/server.key --ssl-certfile /certs/server.crt")


def test_no_port_is_published_beyond_localhost():
    for name, service in services().items():
        for port in service.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), f"{name} publishes {port} on every interface"


def test_dockerfile_uses_python_312_slim():
    text = (REPO / "deploy" / "interviewer.Dockerfile").read_text()
    assert text.splitlines()[0].startswith("FROM python:3.12-slim")


def test_setup_downloads_the_pinned_models_and_pulls_qwen():
    text = SETUP.read_text()
    base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
    assert base in text and "kokoro-v1.0.int8.onnx" in text and "voices-v1.0.bin" in text
    assert "ollama pull" in text and "qwen3:4b-instruct" in text
    assert "manage.py migrate" in text and "make_certs.py" in text


def test_package_contains_the_project(tmp_path):
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("needs the git checkout")
    out = tmp_path / "ai-interviewer.zip"
    result = subprocess.run(["bash", str(REPO / "scripts" / "package.sh"), str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    names = subprocess.run(["unzip", "-Z1", str(out)], capture_output=True, text=True).stdout.split()
    assert "ai-interviewer/README.md" in names and "ai-interviewer/kits/senior-dotnet.yaml" in names
    assert "ai-interviewer/scripts/setup.sh" in names and "ai-interviewer/deploy/docker-compose.yml" in names
    private = ("ai-interviewer/.env", "ai-interviewer/deploy/attendee.env", "ai-interviewer/data/",
               "ai-interviewer/models/", "ai-interviewer/vendor/", "ai-interviewer/certs/", "ai-interviewer/.superpowers/")
    assert not [n for n in names if n.startswith(private) and n != "ai-interviewer/.env.example"]
