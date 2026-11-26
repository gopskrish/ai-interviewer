#!/usr/bin/env bash
# Builds dist/ai-interviewer.zip from the committed files. Nothing private goes in:
# .env, deploy/attendee.env, certs, data and models aren't in git.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/dist/ai-interviewer.zip}"
mkdir -p "$(dirname "$OUT")"
git -C "$ROOT" archive --format=zip --prefix=ai-interviewer/ -o "$OUT" HEAD
echo "Wrote $OUT"
