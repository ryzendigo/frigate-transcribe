# Frigate Transcribe

Continuous audio transcription, speaker diarisation, and end-of-day LLM summaries for one [Frigate NVR](https://frigate.video) camera.

![Web UI](https://img.shields.io/badge/Web_UI-included-blue) ![Docker](https://img.shields.io/badge/Docker-ready-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## Why?

Frigate happily restreams camera audio over RTSP, but nothing in the Frigate ecosystem actually does anything with it. There are short-clip transcription add-ons for Home Assistant, but no project that takes a single Frigate camera and turns its audio into a searchable, labelled, day-by-day transcript on its own.

This is that project.

It runs a self-contained pipeline: pull audio from one Frigate camera, transcribe it with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on CPU, optionally rewrite phonetic-nonsense lines through a fast LLM with rolling conversation context, attribute each segment to a speaker via [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) embeddings, store everything in SQLite with FTS5 full-text search, and push a single end-of-day summary to Pushover.

**Features:**

- Continuous audio capture from any Frigate camera (no event triggering)
- Whisper transcription on CPU, distilled-large-v3 by default (~1.5 GB model, near large-v3 quality, runs at roughly real-time on 8 cores)
- Speaker diarisation with Resemblyzer + nearest-centroid attribution
- Optional LLM correction pass that rewrites obvious phonetic nonsense from Whisper into plausible speech using a rolling context window (Gemini Flash Lite or local Ollama)
- End-of-day LLM summary with TL;DR, highlights, per-speaker recap, mentions of names you care about, pushed to Pushover
- Web UI for browsing day-by-day transcripts, labelling speakers, searching, viewing speaker centroids and unattributed clusters
- Whisper hallucination filtering (no-speech probability, log-prob, compression ratio, dynamic prompt-leak detection)
- Keyword priority alerts via Pushover
- Audio archive with configurable retention; transcripts kept forever
- Optional Uptime Kuma push-monitor heartbeats
- Single Docker container, SQLite-only — no external database

## How it works

```
Frigate RTSP --> ffmpeg (60s chunks)
                     |
                     v
              faster-whisper        Resemblyzer
              (transcribe)          (speaker embedding)
                     |                   |
                     +--------+----------+
                              v
                LLM correction pass (optional, per chunk)
                  - Gemini 2.5 Flash Lite (default), or
                  - local Ollama
                              |
                              v
                  SQLite (segments + speakers + FTS5)
                              |
                              v
              FastAPI UI    End-of-day LLM summary -> Pushover
```

Audio is captured in fixed-length chunks (default 60 s). Each chunk is run through faster-whisper, then optionally rewritten line-by-line by a fast LLM that's given the last ~20 lines of conversation as context. Segments below configurable confidence thresholds (or matching dynamically derived prompt-leak substrings) are dropped.

If speaker embeddings are enabled, each segment's speech section is embedded with Resemblyzer and matched against per-speaker centroids you build by labelling segments through the UI. The match threshold is conservative by default — segments below it are stored as "Unknown" rather than mislabelled.

Once a day at a configurable local time, every segment from the previous day is fed to a separate LLM summary prompt that produces a structured TL;DR / highlights / per-speaker recap / mentions / visitors block, posted to Pushover.

## Quick start

### 1. Bring up the container

```yaml
services:
  frigate-transcribe:
    container_name: frigate-transcribe
    build: https://github.com/ryzendigo/frigate-transcribe.git
    restart: unless-stopped
    environment:
      TRANSCRIBE_CAMERA: dining            # the Frigate camera that has audio
      RTSP_URL: rtsp://frigate:8554/dining # Frigate's go2rtc restream
      TZ: Australia/Perth                  # for daily summary scheduling
      TRANSCRIBE_GEMINI_API_KEY: ${GEMINI_API_KEY}
      TRANSCRIBE_PUSHOVER_TOKEN: ${PUSHOVER_TOKEN}
      PUSHOVER_USER_KEY: ${PUSHOVER_USER}
    volumes:
      - ./data:/data
      - ./models:/models
    ports:
      - "8767:8767"
    mem_limit: 8g
    cpus: 6
```

The container needs to be on the same Docker network as Frigate so it can reach `rtsp://frigate:8554`. If you're running Frigate on a different host, use the host's IP and port 8554.

A first run downloads the Whisper model (~1.5 GB for `distil-large-v3`) into `/models` and then idles waiting for audio chunks. The first transcription takes ~30 s while ffmpeg fills the first chunk.

### 2. Open the UI

```
http://<docker-host>:8767
```

The day view is the default landing page: pick a date, see every segment grouped into conversation blocks, click any segment to play its audio. Use the speakers page to label unattributed clips into named speakers — those labels build per-speaker centroids that future segments are matched against.

### 3. Decide on the LLM correction pass

The correction pass is on by default. It costs roughly one Gemini Flash Lite call per minute of audio. If you don't want to use Gemini, set `TRANSCRIBE_LLM_BACKEND=ollama` and point `OLLAMA_URL` at your Ollama host, or disable it entirely with `TRANSCRIBE_LLM_CORRECTION_ENABLED=0`.

For most setups the correction pass is the difference between a transcript that's worth reading and a transcript that's mostly Whisper guessing.

## Configuration

All configuration is environment variables. Anything not set falls back to a sensible default. The full list with explanations is in [`.env.example`](.env.example), but the ones most worth knowing about:

| Variable | Default | What it does |
| --- | --- | --- |
| `TRANSCRIBE_CAMERA` | `dining` | Frigate camera name (used in default `RTSP_URL` and shown in UI) |
| `RTSP_URL` | `rtsp://frigate:8554/$TRANSCRIBE_CAMERA` | Audio-bearing RTSP stream |
| `TRANSCRIBE_WHISPER_MODEL` | `distil-large-v3` | Any faster-whisper model name |
| `TRANSCRIBE_WHISPER_BEAM` | `5` | Drop to `3` if your CPU is keeping up but only just |
| `TRANSCRIBE_LLM_BACKEND` | `gemini` | Or `ollama` |
| `TRANSCRIBE_LLM_CORRECTION_ENABLED` | `1` | `0` to skip the correction pass entirely |
| `TRANSCRIBE_SPEAKER_NAMES` | empty | Comma-separated names of expected speakers — used to bias Whisper and to remind the summary LLM not to fabricate speaker names |
| `TRANSCRIBE_LOCATION_HINT` | `a home` | Free-form phrase used in the correction prompt — e.g. `a kitchen in Sydney`, `an open-plan office` |
| `TRANSCRIBE_DIALECT_HINT` | `everyday speech` | Free-form phrase used in the correction prompt — e.g. `Australian English`, `American English` |
| `TRANSCRIBE_DIALECT_HINT_WORDS` | empty | Optional comma-separated words to inject into Whisper's initial prompt as accent bias |
| `TRANSCRIBE_MENTION_WATCHLIST` | empty | Comma-separated names highlighted in the UI and given dedicated sections in summaries |
| `TRANSCRIBE_KEYWORD_ALERTS` | empty | Comma-separated keywords that trigger a priority Pushover when heard |
| `TRANSCRIBE_SUMMARY_TIME` | `00:05` | Local-time `HH:MM` for the daily summary |
| `TRANSCRIBE_AUDIO_RETENTION_DAYS` | `90` | Audio chunks older than this are pruned. Transcripts are kept forever. |

## Hardware

CPU-only by design. There is no GPU acceleration anywhere in the pipeline.

A 6-core/12-thread modern x86 CPU runs `distil-large-v3` at roughly real-time on 60 s chunks with `int8` compute and beam=3. If you fall behind real-time the chunk backlog grows and `TRANSCRIBE_MAX_CHUNK_BACKLOG` will eventually start dropping the oldest waiting chunk. Drop to `medium.en` if you have to.

Memory: the Whisper model itself is ~1.5 GB; total container RSS sits around 1.5–2 GB during transcription. Bump `mem_limit` to 8 GB to give Resemblyzer room.

## Data layout

Everything the container produces lives in `/data` inside the container:

```
/data
├── audio/          # 60s WAV chunks, kept for TRANSCRIBE_AUDIO_RETENTION_DAYS
├── chunks/         # raw chunks waiting to be transcribed (transient)
├── logs/           # daily transcription logs
└── transcripts.db  # SQLite — segments, speakers, summaries, FTS5 index
```

`/models` is the Whisper model cache — mount it as a separate volume so a container rebuild doesn't re-download the model.

The SQLite DB is the source of truth. You can `sqlite3 transcripts.db .schema` to see everything; the FTS5 index makes substring search across days fast.

## License

MIT — see [LICENSE](LICENSE).
