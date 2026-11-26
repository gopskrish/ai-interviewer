#!/usr/bin/env bash
# One-time setup for the AI interviewer trial: Ubuntu in WSL2 with Docker Engine.
# Safe to run again: every step that's already done is skipped.
set -euo pipefail

ATTENDEE_REPO="https://github.com/attendee-labs/attendee.git"
ATTENDEE_COMMIT="e7a0b829867b6f6413a0118878a81ddb327b7681"
KOKORO_BASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
KOKORO_FILES=("kokoro-v1.0.int8.onnx" "voices-v1.0.bin")
DEFAULT_OLLAMA_MODEL="qwen3:4b-instruct"
IMAGE="ai-interviewer:local"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

step() { printf '\n==> %s\n' "$*"; }
skip() { printf '    Already done: %s\n' "$*"; }
warn() { printf '    Warning: %s\n' "$*"; }
die() { printf '\nSetup stopped: %s\n' "$*" >&2; exit 1; }
compose() { docker compose -f "$ROOT/deploy/docker-compose.yml" "$@"; }
in_image() { docker run --rm --user "$(id -u):$(id -g)" -v "$ROOT:/work" -w /work "$IMAGE" "$@"; }

step "1/12 Checking docker, git and this PC"
command -v git >/dev/null || die "git is missing. Install it: sudo apt-get install -y git"
command -v curl >/dev/null || die "curl is missing. Install it: sudo apt-get install -y curl"
command -v docker >/dev/null || die "docker is missing. Install Docker Engine first (README, step 3)."
docker compose version >/dev/null 2>&1 || die "the Docker Compose plugin is missing (README, step 3)."
docker info >/dev/null 2>&1 || die "can't reach Docker. Start it (sudo service docker start), and if you just joined \
the docker group, close this terminal and open a new one."
mem_kb=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
if [ "${mem_kb:-0}" -lt 9000000 ]; then
  warn "WSL has only $((mem_kb / 1024 / 1024)) GB of memory. Set memory=10GB in %UserProfile%\\.wslconfig (README, step 2)."
fi
free_kb=$(df -Pk "$ROOT" | awk 'NR == 2 {print $4}')
if [ "${free_kb:-0}" -lt 25000000 ]; then
  warn "only $((free_kb / 1024 / 1024)) GB of disk space is free here; the images and models need about 20 GB."
fi
if [[ "$ROOT" == /mnt/* ]]; then
  warn "the project is on a Windows drive ($ROOT). It runs much faster in your Linux home folder (README, step 4)."
fi

step "2/12 Getting Attendee at commit ${ATTENDEE_COMMIT:0:7}"
if [ ! -d vendor/attendee/.git ]; then
  git clone "$ATTENDEE_REPO" vendor/attendee
fi
if [ "$(git -C vendor/attendee rev-parse HEAD)" = "$ATTENDEE_COMMIT" ]; then
  skip "Attendee is at the pinned commit"
else
  git -C vendor/attendee fetch --quiet origin
  git -C vendor/attendee -c advice.detachedHead=false checkout --quiet "$ATTENDEE_COMMIT"
fi

step "3/12 Building the interviewer image (quick when nothing changed)"
docker build -f deploy/interviewer.Dockerfile -t "$IMAGE" .

step "4/12 Making the local certificates"
mkdir -p certs data models/kokoro models/whisper models/piper
if [ -s certs/server.crt ] && [ -s certs/server.key ] && [ -s certs/ca-bundle.pem ]; then
  skip "certs/ exists"
else
  in_image python scripts/make_certs.py --out certs --hosts interviewer,localhost
fi

step "5/12 Writing Attendee's private settings (deploy/attendee.env)"
if [ -s deploy/attendee.env ]; then
  skip "deploy/attendee.env exists"
else
  generated=$(in_image python -c '
import secrets
from cryptography.fernet import Fernet
user = "attendee" + secrets.token_hex(4)
password = secrets.token_urlsafe(24)
print("CREDENTIALS_ENCRYPTION_KEY=" + Fernet.generate_key().decode())
print("DJANGO_SECRET_KEY=" + secrets.token_urlsafe(50))
print("AWS_ACCESS_KEY_ID=" + user)
print("AWS_SECRET_ACCESS_KEY=" + password)
print("MINIO_ROOT_USER=" + user)
print("MINIO_ROOT_PASSWORD=" + password)
')
  (
    umask 077
    {
      echo "# Made by scripts/setup.sh. Keep it private; it holds Attendee's keys."
      echo "$generated"
      echo "AWS_DEFAULT_REGION=us-east-1"
      echo "AWS_ENDPOINT_URL=http://minio:9000"
      echo "AWS_RECORDING_STORAGE_BUCKET_NAME=attendee-recordings"
    } > deploy/attendee.env
  )
fi

step "6/12 Creating .env"
if [ -f .env ]; then
  skip ".env exists"
else
  (umask 077 && cp .env.example .env)
fi

step "7/12 Downloading the Kokoro voice"
for name in "${KOKORO_FILES[@]}"; do
  if [ -s "models/kokoro/$name" ]; then
    skip "$name"
  else
    curl -fL --retry 3 --progress-bar -o "models/kokoro/$name.part" "$KOKORO_BASE$name"
    mv "models/kokoro/$name.part" "models/kokoro/$name"
  fi
done

step "8/12 Starting the stack (the first run builds Attendee, which takes a while)"
compose up -d

step "9/12 Setting up Attendee's database"
migrated=""
for _ in $(seq 1 30); do
  if compose exec -T attendee-app python manage.py migrate --noinput; then
    migrated=yes
    break
  fi
  sleep 5
done
[ -n "$migrated" ] || die "Attendee's database setup failed. Look at: docker compose logs attendee-app postgres"

step "10/12 Downloading the language model"
model=$(sed -n 's/^OLLAMA_MODEL=//p' .env | tail -n 1)
model="${model:-$DEFAULT_OLLAMA_MODEL}"
compose exec -T ollama ollama pull "$model"

step "11/12 Downloading the Whisper speech model"
compose run --rm --no-deps -T interviewer python -c '
import asyncio
from interviewer.ai.whisper_stt import FasterWhisperSTT
from interviewer.config import Settings
s = Settings.from_env()
asyncio.run(FasterWhisperSTT(s.whisper_model, s.whisper_cpu_threads, download_root=s.models_dir / "whisper").warm_up())
print("Whisper", s.whisper_model, "is ready")
'

step "12/12 Done. What's left is manual:"
cat <<EOF

  1. Open http://localhost:8000 in Windows and sign up for Attendee.
     The confirmation link is printed in the log; find it with:
       cd $ROOT/deploy && docker compose logs attendee-app | grep -i confirm
  2. In Attendee, open API Keys, create a key, and put it in $ROOT/.env:
       ATTENDEE_API_KEY=<the key>
  3. Restart the interviewer so it picks up the key:
       cd $ROOT/deploy && docker compose up -d interviewer
  4. Open https://localhost:8443 and run the day-one check (/check) first.
     Optional: trust certs/ca.crt in Windows to skip the browser warning (README).

Trial rule: interview colleagues only, never real candidates.
EOF
