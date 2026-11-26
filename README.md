# AI Interviewer (free trial version)

An AI that runs first-round technical interviews in Microsoft Teams. It joins the meeting as a guest called
"AI Interviewer", says it's an AI and that the session is recorded, asks for consent, then runs a 30-minute
interview from a kit (the included kit is for a **Senior .NET developer**). Afterwards you get a transcript and
a scorecard with a 1–4 score per topic and quoted evidence. It never makes a hire or reject call; a person
reads the evidence and decides.

Everything runs on one Windows PC with free, local models: no paid services, and the audio never leaves the PC
except into Teams.

| Part | What it does |
| --- | --- |
| [Attendee](https://github.com/attendee-labs/attendee) (self-hosted) | Joins Teams as a guest in headless Chrome and streams the call audio |
| faster-whisper `small` | Speech to text |
| Ollama with `qwen3:4b-instruct` | Decides follow-ups and scores the interview |
| Kokoro (Piper as a fallback) | Text to speech |
| `interviewer` (this project) | Dashboard, practice page, interview logic and scoring at https://localhost:8443 |

> **Trial rules.** Interview colleagues only, never real candidates, until the trial is done and the hosted
> version is set up. Attendee is under the Elastic License 2.0: fine for internal use, not for offering as a
> service to others.

## What you need

- Windows 10 (22H2) or 11, with 16 GB of RAM and about 25 GB of free disk space. No graphics card is needed.
- Admin rights once, to install WSL2.
- A Teams meeting where guests (anonymous users) are allowed to join.
- An hour for the first setup; most of it is downloading and building.

## Setup

### 1. Install Ubuntu in WSL2

In PowerShell **as administrator**:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart if asked, then open "Ubuntu 24.04" from the Start menu and create your Linux user.

### 2. Give WSL enough memory

By default WSL gets half your RAM (8 GB), which is too little. Create the file `%UserProfile%\.wslconfig`
(for example `C:\Users\you\.wslconfig`) with:

```ini
[wsl2]
memory=10GB
```

Then, in PowerShell: `wsl --shutdown`, and open Ubuntu again. That leaves about 6 GB for Windows, Teams and a
browser.

### 3. Install Docker Engine in Ubuntu

In the Ubuntu terminal:

```bash
sudo apt-get update && sudo apt-get install -y curl git unzip
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

Close the Ubuntu window and open a new one (so the docker group applies), then check with `docker run --rm hello-world`.
If Docker isn't running, start it with `sudo service docker start`.

### 4. Copy the project into your Linux home folder and run setup

Keep the project in the Linux home folder, not on `C:` (`/mnt/c/...`): Docker is much slower there.
If the zip is in your Windows Downloads folder:

```bash
cp /mnt/c/Users/<your Windows user>/Downloads/ai-interviewer.zip ~
cd ~ && unzip ai-interviewer.zip && cd ai-interviewer
bash scripts/setup.sh
```

`setup.sh` can be run again at any time; it skips whatever is already done. It:

1. checks Docker, git, memory and disk space;
2. downloads Attendee and pins it to commit `e7a0b82`;
3. builds the interviewer image;
4. makes a local certificate authority and the server certificate (`certs/`);
5. writes Attendee's private keys to `deploy/attendee.env`;
6. creates `.env` from `.env.example`;
7. downloads the Kokoro voice into `models/kokoro/`;
8. starts the stack (`docker compose up -d`);
9. sets up Attendee's database;
10. downloads the language model (`qwen3:4b-instruct`, about 2.5 GB);
11. downloads the Whisper speech model;
12. prints the manual steps below.

### 5. Connect Attendee

1. Open http://localhost:8000 in your Windows browser and sign up.
2. Attendee "sends" the confirmation email to its log. Find the link with:
   ```bash
   cd ~/ai-interviewer/deploy && docker compose logs attendee-app | grep -i confirm
   ```
   Open it, then log in.
3. Open **API Keys**, create a key and copy it.
4. Put it in `~/ai-interviewer/.env`:
   ```
   ATTENDEE_API_KEY=<the key>
   ```
5. Restart the interviewer: `cd ~/ai-interviewer/deploy && docker compose up -d interviewer`.

### 6. (Optional) Trust the certificate in Windows

Without this, the browser warns once that https://localhost:8443 isn't trusted, and you click through.
To trust it instead, copy the certificate to Windows:

```bash
cp ~/ai-interviewer/certs/ca.crt /mnt/c/Users/<your Windows user>/Downloads/
```

and in a Command Prompt **as administrator**:

```bat
certutil -addstore -f ROOT %UserProfile%\Downloads\ca.crt
```

Restart the browser. Edge and Chrome use the Windows store. Only `ca.crt` is meant to leave WSL: never copy
`ca.key` anywhere.

## First run: the day-one check

Open https://localhost:8443 and go to **Day-one check**. Do this before anything else.

1. **Echo test.** Create a Teams meeting, set *Meeting options → Who can bypass the lobby* to *Everyone*, join it
   yourself, and paste its link into the check page. When the bot is in the call, say hello. It greets you and
   then repeats each sentence you say. It passes when:
   - the bot gets into the call (or into the lobby, where you admit it);
   - you hear the greeting in Teams;
   - what you say shows up as text on the page;
   - the bot's own voice does **not** show up as text (the page warns if it does).
2. **Speed test.** Click *Run speed test*. It times speech-to-text, one interview decision and text-to-speech on
   this PC, against the target of replying within 8 seconds of the candidate stopping. To compare models, pull
   them first and list them, comma-separated:
   ```bash
   cd ~/ai-interviewer/deploy
   docker compose exec ollama ollama pull llama3.2:3b
   docker compose exec ollama ollama pull gemma3:4b
   ```
   The same test runs from the terminal:
   `docker compose exec interviewer python -m interviewer.daycheck bench --models qwen3:4b-instruct,llama3.2:3b,gemma3:4b`.
   Pick the fastest model that passes, set `OLLAMA_MODEL` in `.env`, and run `docker compose up -d interviewer`.

## Practice interview

**Practice** on the dashboard runs the same interviewer with your browser's microphone; nothing goes to Teams.
Use headphones, click *Start microphone* and say hello. The session page shows the transcript, and the scorecard
a few minutes after you finish.

## Running a trial interview

1. Create a Teams meeting and invite the colleague. In *Meeting options*, set *Who can bypass the lobby* to
   *Everyone*, or be ready to admit "AI Interviewer" from the lobby.
2. On the dashboard click **New interview**: pick the kit, type the colleague's name, paste the meeting link.
   Resume notes are optional; with them, the two experience questions are tailored (starting then takes up to a
   minute). Click **Start**.
3. The session page shows the bot's status: joining, in the lobby, in the call.
4. The interview begins when you click **Begin**, or by itself when someone speaks. The AI says it's an AI and
   that the session is recorded, and asks "Is that OK with you?". If the answer is no, it says goodbye and leaves.
5. If you watch the interview, **stay muted**. The bot hears everyone in the call as one voice, so anything you
   say counts as the candidate's answer.
6. **End interview** stops it early: the AI says goodbye and the bot leaves. Otherwise it ends by itself after
   about 30 minutes.
7. Scoring takes a few minutes on CPU (the page shows "Scoring…"). Then the scorecard appears, with Markdown and
   transcript downloads. If another session is live, scoring waits for it so the two don't compete for the CPU.

To post scorecards to a Teams channel, create a webhook with the Teams Workflows app ("Post to a channel when a
webhook request is received") and put its URL in `TEAMS_WEBHOOK_URL` in `.env`. Practice scorecards are never
posted.

## Changing the interview

The kit is `kits/senior-dotnet.yaml`: the intro and closing, warm-up and experience questions, the topic bank
with rubrics, which three topics are used, answers to common candidate questions, forbidden topics and
vocabulary hints for speech recognition. Edit it and reload the dashboard: kits are re-read on each page load,
and a kit with a mistake is listed with the line number. A session that is already running keeps the kit it
started with. To add a role, copy the file under a new name.

## Settings

All settings are in `.env` (see `.env.example`). After changing one, run `docker compose up -d interviewer`
from the `deploy` folder.

| Setting | Default | What it's for |
| --- | --- | --- |
| `OLLAMA_MODEL` | `qwen3:4b-instruct` | The language model. Pull it first with `ollama pull`. |
| `LLM_TIMEOUT_S` | `20` | How long a decision may take before the interviewer falls back to a simple rule. |
| `WHISPER_MODEL` | `small` | `base` is faster and less accurate. |
| `WHISPER_CPU_THREADS` | `4` | CPU threads for speech recognition. |
| `TTS_ENGINE` | `kokoro` | `piper` is faster on slow PCs (download the voice first, see below). |
| `KOKORO_VOICE` | `af_heart` | The Kokoro voice. |
| `ATTENDEE_API_KEY` | (empty) | From the Attendee UI. Teams interviews are off without it; practice works. |
| `TEAMS_WEBHOOK_URL` | (empty) | Optional: where scorecards are posted. |
| `END_OF_TURN_SILENCE_S` | `1.5` | Silence that ends an answer (short answers get 3 s, so people can think). |
| `RETENTION_DAYS` | `30` | Sessions older than this are deleted, daily. |
| `AI_MODE` | `local` | `openai` is planned for the hosted version and isn't built yet. |

To use Piper, download a voice into `models/piper/` and set `TTS_ENGINE=piper`:

```bash
cd ~/ai-interviewer/models/piper
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

## Day to day

Run these from `~/ai-interviewer/deploy`:

| To | Run |
| --- | --- |
| Start everything | `docker compose up -d` |
| Stop everything (frees the memory) | `docker compose stop` |
| See the interviewer's log | `docker compose logs -f interviewer` |
| See the bot's log | `docker compose logs -f attendee-worker` |
| Delete old sessions now | `docker compose exec interviewer python -m interviewer.cleanup --days 30` |

Each session lives in `data/sessions/<id>/`: the transcript, the scorecard and the call audio
(`meeting-audio.wav`). They're deleted after `RETENTION_DAYS`.

## Troubleshooting

**The bot is stuck in the lobby.** Admit "AI Interviewer" in Teams, or set *Who can bypass the lobby* to
*Everyone* before starting. If guests can't join meetings at all in your organisation, the bot can't either;
that's a Teams admin setting. After 3 minutes the session page shows the bot's last status and the bot is told
to leave.

**The bot is in the call but nothing happens.** The session page warns when the bot's audio connection doesn't
arrive. That's almost always the certificate: run `bash scripts/setup.sh` again (it recreates missing
certificates), then `docker compose up -d`. Look for `BotWebsocketClient` in `docker compose logs attendee-worker`.

**Replies are slow, or the AI says "Thanks" and then takes long.** Run the speed test. Close other heavy apps,
try a smaller model (`llama3.2:3b`), set `WHISPER_MODEL=base`, or switch to Piper. If the language model doesn't
answer within `LLM_TIMEOUT_S`, the interviewer moves on by a simple rule rather than waiting, so the interview
continues, with fewer tailored follow-ups.

**Not enough memory** (containers restart, Ubuntu freezes, `docker compose logs` shows "killed"). Check
`.wslconfig` (step 2) and `free -h` in Ubuntu. Close browser tabs and other apps during interviews, and stop the
stack when you're not using it (`docker compose stop`).

**The browser blocks the microphone on the practice page.** Allow the microphone for https://localhost:8443 in
the site settings. The page must be opened as https://localhost:8443, not by IP address.

**"There's no usable interview kit".** A kit has a mistake: the dashboard lists the file and the line.

## Later: hosted and paid

The design (`docs/superpowers/specs/2026-11-25-ai-interviewer-design.md`) covers the hosted phase: running on a
server or your organisation's Teams setup, real candidates, and `AI_MODE=openai` for faster and better speech
and language models. The trial build keeps those parts out on purpose.

## For developers

```bash
python3.12 -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]'
python -m pytest
node --test tests/js/*.test.mjs
bash scripts/package.sh   # writes dist/ai-interviewer.zip from the committed files
```

The tests use fake speech and language models, so they need no downloads.
