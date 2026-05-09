"""Continuously transcribe an RTSP audio feed into a searchable SQLite database,
per-segment speaker embeddings for attribution, and run a single end-of-day
summary via an LLM. Exposes a FastAPI UI for browsing and labelling."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import pathlib
import re
import shutil
import signal
import subprocess
import threading
import time
from collections import deque

import httpx
import numpy as np
import soundfile as sf
import uvicorn
from faster_whisper import WhisperModel

from db import Database
from embedder import Embedder
import summariser
import ui as ui_module


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n)
        if v not in (None, ""):
            return v
    return default


CAMERA_NAME = _env("TRANSCRIBE_CAMERA", "CAMERA_NAME", default="dining")
RTSP_URL = _env("RTSP_URL", default=f"rtsp://frigate:8554/{CAMERA_NAME}")
# distil-large-v3: distilled Large v3, ~1.5GB, ~medium.en speed on CPU,
# quality close to large-v3 and significantly better on non-US accents than
# medium.en. Override with TRANSCRIBE_WHISPER_MODEL.
MODEL_SIZE = _env("TRANSCRIBE_WHISPER_MODEL", "WHISPER_MODEL", default="distil-large-v3")
COMPUTE_TYPE = _env("TRANSCRIBE_WHISPER_COMPUTE", "WHISPER_COMPUTE_TYPE", default="int8")
CPU_THREADS = int(_env("TRANSCRIBE_WHISPER_THREADS", "WHISPER_CPU_THREADS", default="8"))
BEAM_SIZE = int(_env("TRANSCRIBE_WHISPER_BEAM", "WHISPER_BEAM_SIZE", default="5"))
LANGUAGE = _env("TRANSCRIBE_WHISPER_LANGUAGE", "WHISPER_LANGUAGE", default="en")
# 500ms silence before splitting — 300ms over-fragments natural speech pauses
# and strips context from Whisper, which tanks accuracy on short utterances.
VAD_MIN_SILENCE_MS = int(_env("TRANSCRIBE_VAD_MIN_SILENCE_MS", default="500"))
NO_SPEECH_THRESHOLD = float(_env("TRANSCRIBE_WHISPER_NO_SPEECH_THRESHOLD",
                                 default="0.6"))
LOG_PROB_THRESHOLD = float(_env("TRANSCRIBE_WHISPER_LOG_PROB_THRESHOLD",
                                default="-1.0"))
# Drop segments with avg log-prob below this — confidence-based filter that
# catches mumbled / noisy Whisper output the other heuristics miss.
AVG_LOGPROB_MIN = float(_env("TRANSCRIBE_AVG_LOGPROB_MIN", default="-0.9"))
# Drop segments with suspiciously high text-vs-audio compression ratio. Whisper
# loops/repeats produce very high ratios; natural speech sits around 1.2–1.8.
COMPRESSION_RATIO_MAX = float(_env("TRANSCRIBE_COMPRESSION_RATIO_MAX",
                                   default="2.4"))
# Audio pipeline: afftdn removes fan/AC broadband noise, loudnorm gives Whisper
# a consistent LUFS level without the noise-floor-pumping side-effects of
# dynaudnorm. highpass trims sub-100Hz rumble (HVAC, plate scrapes). Previous
# "highpass=80,dynaudnorm=p=0.95:g=15" was too aggressive and amplified room
# noise to levels that caused phonetic errors (e.g. "day" → "dad").
AUDIO_FILTER = _env("TRANSCRIBE_AUDIO_FILTER",
                    default="highpass=f=80,"
                            "loudnorm=I=-18:TP=-1.5:LRA=11")
if AUDIO_FILTER.strip().lower() in {"none", "off", "disabled"}:
    AUDIO_FILTER = ""
# Optional LLM post-editing pass — sends each chunk's raw Whisper segments
# to a fast LLM (default: Gemini Flash Lite) with rolling conversation
# context and asks it to rewrite obviously mis-heard phonetic-nonsense lines.
# Falls back to raw output on ANY failure, so the main pipeline stays robust.
LLM_CORRECTION_ENABLED = _env("TRANSCRIBE_LLM_CORRECTION_ENABLED",
                              default="1").lower() not in {"0", "false", "no"}
LLM_CORRECTION_MODEL = _env("TRANSCRIBE_LLM_CORRECTION_MODEL",
                            default="gemini-2.5-flash-lite")
LLM_CORRECTION_CONTEXT_LINES = int(_env("TRANSCRIBE_LLM_CORRECTION_CONTEXT_LINES",
                                        default="20"))
DATA_DIR = pathlib.Path(_env("DATA_DIR", default="/data"))
MODELS_DIR = pathlib.Path(_env("MODELS_DIR", default="/models"))
CHUNKS_DIR = DATA_DIR / "chunks"
AUDIO_DIR = DATA_DIR / "audio"
LOGS_DIR = DATA_DIR / "logs"
DB_PATH = DATA_DIR / "transcripts.db"
CHUNK_SECONDS = int(_env("TRANSCRIBE_CHUNK_SECONDS", "CHUNK_SECONDS", default="60"))
SUMMARY_TIME = _env("TRANSCRIBE_SUMMARY_TIME", "SUMMARY_TIME", default="00:05")
TZ_NAME = _env("TZ", default="UTC")
UI_HOST = _env("UI_HOST", default="0.0.0.0")
UI_PORT = int(_env("TRANSCRIBE_UI_PORT", "UI_PORT", default="8767"))
UI_BASE_URL = _env("TRANSCRIBE_UI_BASE_URL", "UI_BASE_URL", default="").strip()
SPEAKER_MATCH_THRESHOLD = float(_env("TRANSCRIBE_SPEAKER_MATCH_THRESHOLD",
                                     "SPEAKER_MATCH_THRESHOLD", default="0.78"))
AUDIO_RETENTION_DAYS = int(_env("TRANSCRIBE_AUDIO_RETENTION_DAYS",
                                "AUDIO_RETENTION_DAYS", default="90"))
ENABLE_EMBEDDINGS = _env("TRANSCRIBE_ENABLE_EMBEDDINGS", "ENABLE_EMBEDDINGS",
                         default="1").lower() not in {"0", "false", "no"}
NO_SPEECH_PROB_MAX = float(_env("TRANSCRIBE_NO_SPEECH_PROB_MAX",
                                default="0.55"))
LANGUAGE_PROB_MIN = float(_env("TRANSCRIBE_LANGUAGE_PROB_MIN", default="0.0"))
MAX_CHUNK_BACKLOG = int(_env("TRANSCRIBE_MAX_CHUNK_BACKLOG", default="30"))
FFMPEG_STALL_SECONDS = int(_env("TRANSCRIBE_FFMPEG_STALL_SECONDS",
                                default=str(CHUNK_SECONDS * 3)))
UPTIME_KUMA_PUSH_URL = _env("TRANSCRIBE_UPTIME_KUMA_PUSH_URL", default="")
DISK_FREE_MIN_GB = float(_env("TRANSCRIBE_DISK_FREE_MIN_GB", default="5"))
# Comma-separated names of expected speakers. Used to bias Whisper's
# recognition (via the initial prompt) and to remind the LLM not to
# fabricate speaker attributions in summaries. Leave blank if you don't
# have a stable set of speakers.
SPEAKER_NAMES = [
    n.strip() for n in _env("TRANSCRIBE_SPEAKER_NAMES", default="").split(",")
    if n.strip()
]
# Free-form phrase describing where the audio is captured. Used in the
# LLM correction prompt to give the model context. Examples:
#   "a kitchen in Sydney"  "an open-plan office"  "a small radio studio"
LOCATION_HINT = _env("TRANSCRIBE_LOCATION_HINT", default="a home")
# Free-form phrase describing the expected dialect / register, e.g.:
#   "Australian English"  "American English"  "British English with industry jargon"
DIALECT_HINT = _env("TRANSCRIBE_DIALECT_HINT", default="everyday speech")
# Optional comma-separated dialect hint words to bias Whisper away from
# substitution errors typical of US-centric models. Examples for AU:
#   "yeah nah,heaps,reckon,mate,fair enough,alright"
DIALECT_HINT_WORDS = [
    w.strip() for w in _env("TRANSCRIBE_DIALECT_HINT_WORDS", default="").split(",")
    if w.strip()
]


def _default_initial_prompt() -> str:
    """Build a Whisper initial-prompt biasing string from configured names
    and dialect hint words. Deliberately NOT a natural sentence — Whisper
    will echo a coherent grammatical prompt as transcript output when audio
    is silent or noisy. Plain vocabulary fragments bias recognition without
    giving the model a sentence to repeat."""
    parts: list[str] = []
    if SPEAKER_NAMES:
        parts.append(", ".join(SPEAKER_NAMES) + ".")
    if DIALECT_HINT_WORDS:
        parts.append(", ".join(DIALECT_HINT_WORDS) + ".")
    return " ".join(parts)


WHISPER_INITIAL_PROMPT = _env(
    "TRANSCRIBE_WHISPER_INITIAL_PROMPT", default=_default_initial_prompt(),
)
KEYWORD_ALERTS = [
    kw.strip().lower() for kw in _env("TRANSCRIBE_KEYWORD_ALERTS", default="").split(",")
    if kw.strip()
]
KEYWORD_ALERT_COOLDOWN_S = int(_env("TRANSCRIBE_KEYWORD_ALERT_COOLDOWN", default="300"))

# Whisper is known to emit canned transcripts over silence or low-signal audio.
# These substrings, matched case-insensitively after a loose normalisation, are
# treated as artefacts rather than speech and dropped from the DB + summaries.
HALLUCINATION_PHRASES = {
    # YouTube/vlog canned endings
    "thanks for watching", "thank you for watching", "thank you for watching.",
    "see you in the next video", "see you next video", "see you next time",
    "like and subscribe", "please like and subscribe", "don't forget to subscribe",
    "please subscribe", "subscribe to", "subscribe", "subscribe!",
    "hit the bell icon", "i'll see you next time", "i'll see you in the next one",
    # Single-word noise
    "thank you", "thank you.", "thanks", "thanks.", "thanks!",
    "yes", "yes.", "okay", "okay.", "ok", "ok.",
    "mm-hmm", "mm-hmm.", "mm", "mm.", "uh-huh", "uh-huh.",
    "bye", "bye.", "hi", "hi.", "hello", "hello.", "you", "you.",
    # Ellipsis / punctuation only
    "...", ".", "-", "—",
    # Known whisper artefacts
    "we will see you in the next video",
    "all right",
    "mbc mbc", "mbc",
    "see you", "see you.",
}

def _derive_prompt_leak_substrings(prompt: str) -> tuple[str, ...]:
    """Build leakage detector substrings from the configured initial prompt.

    Whisper occasionally echoes its conditioning prompt as transcript output
    when audio is silent or low-signal. Any 3+ consecutive words taken from
    the user's actual prompt is therefore a strong "this is a hallucination"
    signal — much more reliable than maintaining a hand-curated list."""
    if not prompt:
        return ()
    words = re.findall(r"[\w']+", prompt.lower())
    spans: set[str] = set()
    for n in (3, 4, 5):
        for i in range(len(words) - n + 1):
            spans.add(" ".join(words[i:i + n]))
    return tuple(sorted(spans, key=len, reverse=True))


# Initial-prompt leakage substrings — derived from the active initial prompt
# so leak detection automatically tracks whatever the user has configured.
PROMPT_LEAK_SUBSTRINGS = _derive_prompt_leak_substrings(WHISPER_INITIAL_PROMPT)


_last_chunk_mtime_lock = threading.Lock()
_last_chunk_mtime: float = 0.0
_ffmpeg_pid_lock = threading.Lock()
_ffmpeg_pid: int | None = None


def set_ffmpeg_pid(pid: int | None) -> None:
    with _ffmpeg_pid_lock:
        global _ffmpeg_pid
        _ffmpeg_pid = pid


def touch_last_chunk(mtime: float) -> None:
    with _last_chunk_mtime_lock:
        global _last_chunk_mtime
        if mtime > _last_chunk_mtime:
            _last_chunk_mtime = mtime


def get_last_chunk_age() -> float:
    with _last_chunk_mtime_lock:
        if _last_chunk_mtime == 0.0:
            return 0.0
        return time.time() - _last_chunk_mtime


# Rolling in-memory window of the last N corrected lines — gives Gemini
# enough conversational context to rewrite mis-heard lines. Resets on
# container restart; that's acceptable, the correction just starts cold.
_recent_corrected_lock = threading.Lock()
_recent_corrected: deque[str] = deque(maxlen=80)


def _recent_context_snapshot() -> list[str]:
    with _recent_corrected_lock:
        return list(_recent_corrected)


def _extend_recent_context(lines: list[str]) -> None:
    with _recent_corrected_lock:
        for line in lines:
            line = (line or "").strip()
            if line:
                _recent_corrected.append(line)


def llm_correct_lines(lines: list[str]) -> list[str]:
    """Post-edit raw Whisper segments via a fast LLM with conversation
    context. Returns a list of the same length — on ANY failure (disabled,
    no API key, network, malformed JSON, length mismatch) it returns the
    input unchanged, so the transcription pipeline never blocks on this."""
    if not LLM_CORRECTION_ENABLED or not lines:
        return lines
    api_key = getattr(summariser, "GEMINI_API_KEY", "")
    if not api_key:
        return lines
    ctx_lines = _recent_context_snapshot()
    ctx_block = ""
    if ctx_lines:
        tail = ctx_lines[-LLM_CORRECTION_CONTEXT_LINES:]
        ctx_block = (
            "Recent prior lines from this same conversation (context only — "
            "do NOT include in your output):\n"
            + "\n".join(f"- {l}" for l in tail)
            + "\n\n"
        )
    names_clause = (
        f"Names that may appear: {', '.join(SPEAKER_NAMES)}. "
        if SPEAKER_NAMES else ""
    )
    prompt = (
        "You are aggressively correcting a low-quality speech-to-text "
        f"transcript captured by a distant microphone in {LOCATION_HINT}. "
        "Whisper frequently produces phonetic nonsense from overlapping "
        "speech and unclear audio. Your job is to rewrite those nonsense "
        "lines into what was MOST PLAUSIBLY actually said, using "
        f"{DIALECT_HINT} and the conversation context.\n\n"
        f"{names_clause}Casual swearing is normal; preserve it.\n\n"
        "TREAT THESE AS NONSENSE TO REWRITE:\n"
        "- Phrases that don't make literal sense: rewrite to a plausible "
        "utterance whose phonemes match the raw line.\n"
        "- Grammatically broken output: rewrite to plausible speech.\n"
        "- Wrong-word substitutions that clash with the surrounding "
        "context: e.g. 'poor little shoot' in a swearing register is "
        "almost certainly 'poor little shit'.\n"
        "- Homophones Whisper confuses: 'wood' / 'would', 'tape' / "
        "'take', 'be' / 'been'.\n\n"
        "LEAVE UNCHANGED:\n"
        "- Short confident utterances that already parse ('Yeah.', "
        "'I don't know.').\n"
        "- Any line you're >80% sure is already correct.\n\n"
        "HARD RULES:\n"
        "- Output a JSON array of strings with EXACTLY the same length as "
        "input (one output line per input line, same order).\n"
        "- NEVER add facts not implied by the raw line. If you can't "
        "guess, preserve the literal phonetics as a last resort.\n"
        "- Preserve swearing; don't sanitise.\n"
        "- No preamble, no markdown, no commentary — ONLY the JSON "
        "array.\n\n"
        + ctx_block
        + "Raw lines to post-edit (JSON array of "
        + str(len(lines)) + " strings):\n"
        + json.dumps(lines, ensure_ascii=False)
    )
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{LLM_CORRECTION_MODEL}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json",
            # Gemini 2.5 models default to reasoning ("thinking") mode,
            # which blows past our per-chunk latency budget. We need
            # answers fast, not slow-and-thoughtful.
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    try:
        with httpx.Client(timeout=40.0) as c:
            r = c.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        corrected = json.loads(text)
    except Exception:
        log.warning("llm_correct_lines: API/parse failure, keeping raw", exc_info=True)
        return lines
    if not isinstance(corrected, list) or len(corrected) != len(lines):
        log.warning("llm_correct_lines: length mismatch %d->%s, keeping raw",
                    len(lines),
                    len(corrected) if isinstance(corrected, list) else type(corrected).__name__)
        return lines
    out: list[str] = []
    for orig, cand in zip(lines, corrected):
        cand_str = str(cand).strip() if cand is not None else ""
        # Blank candidate or over-elaboration → keep original.
        if not cand_str:
            out.append(orig)
            continue
        if len(cand_str) > max(len(orig) * 3 + 40, 240):
            out.append(orig)
            continue
        out.append(cand_str)
    changed = sum(1 for a, b in zip(lines, out) if a != b)
    if changed:
        log.debug("llm_correct_lines: rewrote %d/%d lines", changed, len(lines))
    return out

_keyword_alert_state: dict[str, float] = {}
_keyword_alert_lock = threading.Lock()


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(",.!?")


def is_hallucination(
    text: str,
    no_speech_prob: float | None,
    prev_text: str | None,
    avg_logprob: float | None = None,
    compression_ratio: float | None = None,
) -> bool:
    """Return True if the Whisper segment looks like a hallucinated artefact."""
    t = _normalise(text)
    if not t or len(t) < 2:
        return True
    if no_speech_prob is not None and no_speech_prob > NO_SPEECH_PROB_MAX:
        return True
    # Low-confidence decode — strong signal for garbled / noise output.
    if avg_logprob is not None and avg_logprob < AVG_LOGPROB_MIN:
        return True
    # High compression ratio → text is very repetitive → Whisper loop artefact.
    if compression_ratio is not None and compression_ratio > COMPRESSION_RATIO_MAX:
        return True
    if t in HALLUCINATION_PHRASES:
        return True
    # Initial-prompt leakage — Whisper sometimes emits its prompt verbatim.
    for leak in PROMPT_LEAK_SUBSTRINGS:
        if leak in t:
            return True
    # No alphabetic characters at all (punctuation-only line)
    if not any(ch.isalpha() for ch in t):
        return True
    # Very repetitive word distribution (loop artefact)
    words = t.split()
    if len(words) >= 5 and (len(set(words)) / len(words)) < 0.3:
        return True
    # Single short word repeated with trailing "." — e.g. "you. you. you."
    if len(words) >= 3 and len(set(w.rstrip('.,!?') for w in words)) == 1:
        return True
    # Immediate duplicate of previous segment (whisper loops in silence)
    if prev_text is not None and _normalise(prev_text) == t:
        return True
    return False


def fire_keyword_alerts(text: str, start_ts: dt.datetime, segment_id: int,
                        speaker: str | None) -> None:
    if not KEYWORD_ALERTS:
        return
    lower = text.lower()
    now = time.monotonic()
    for kw in KEYWORD_ALERTS:
        if not re.search(rf"\b{re.escape(kw)}\b", lower):
            continue
        with _keyword_alert_lock:
            last = _keyword_alert_state.get(kw, 0.0)
            if now - last < KEYWORD_ALERT_COOLDOWN_S:
                continue
            _keyword_alert_state[kw] = now
        speaker_label = speaker or "Unknown"
        url = None
        if UI_BASE_URL:
            url = (f"{UI_BASE_URL.rstrip('/')}/day/"
                   f"{start_ts.date().isoformat()}#segment-{segment_id}")
        summariser.pushover_notify(
            f"[{CAMERA_NAME}] Heard \u201C{kw}\u201D",
            f"{start_ts.strftime('%H:%M:%S')} {speaker_label}: {text}",
            priority=1, url=url, url_title="Open in transcript" if url else None,
        )
        log.info("keyword alert fired for %r (segment #%d)", kw, segment_id)

for _d in (LOGS_DIR, CHUNKS_DIR, AUDIO_DIR, MODELS_DIR, DATA_DIR / "summaries"):
    _d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("main")


def run_ffmpeg() -> None:
    """Segment RTSP audio into 16kHz mono WAV chunks indefinitely."""
    backoff = 5
    fast_fail_count = 0
    crashloop_alerted = False
    while True:
        pattern = str(CHUNKS_DIR / "chunk_%Y-%m-%dT%H-%M-%S.wav")
        # Reset the watchdog's staleness clock before each spawn, otherwise the
        # "no new chunks for Ns" counter accumulates across restarts and kills
        # each subsequent ffmpeg faster than the last — which trips the 3-in-5s
        # crashloop detector on otherwise-healthy restarts.
        touch_last_chunk(time.time())
        # Note: no -stimeout / -rw_timeout / -timeout — they aren't recognised
        # by the ffmpeg build in this image. The ffmpeg_watchdog thread handles
        # the "stuck stream" case by killing the process.
        # AUDIO_FILTER runs a high-pass + dynaudnorm so distant-mic dining
        # audio reaches Whisper at a usable, consistent loudness.
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-rtsp_transport", "tcp",
            "-i", RTSP_URL,
            "-vn",
        ]
        if AUDIO_FILTER:
            cmd += ["-af", AUDIO_FILTER]
        cmd += [
            "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
            "-f", "segment", "-segment_time", str(CHUNK_SECONDS),
            "-reset_timestamps", "1", "-strftime", "1",
            pattern,
        ]
        log.info("ffmpeg start: %s", " ".join(cmd))
        started = time.monotonic()
        try:
            proc = subprocess.Popen(cmd)
        except Exception:
            log.exception("ffmpeg failed to spawn")
            time.sleep(backoff)
            continue
        set_ffmpeg_pid(proc.pid)
        try:
            rc = proc.wait()
        except Exception:
            log.exception("ffmpeg wait crashed")
            rc = -1
        set_ffmpeg_pid(None)
        duration = time.monotonic() - started
        if rc != 0 and duration < 5:
            fast_fail_count += 1
        else:
            fast_fail_count = 0
            crashloop_alerted = False
        if fast_fail_count >= 3 and not crashloop_alerted:
            # Capture is broken and will keep looping — page the user.
            msg = (f"ffmpeg exited rc={rc} in {duration:.1f}s three times in a row. "
                   f"Audio capture is stopped. Check container logs.")
            log.error("CRASHLOOP: %s", msg)
            try:
                summariser.pushover_notify(
                    f"[{CAMERA_NAME}] transcribe: ffmpeg crashloop", msg, priority=1,
                )
            except Exception:
                log.exception("failed to send crashloop pushover")
            crashloop_alerted = True
        backoff = 5 if duration > 120 else min(backoff * 2, 60)
        log.warning("ffmpeg exited rc=%s after %.0fs, restarting in %ds",
                    rc, duration, backoff)
        time.sleep(backoff)


def ffmpeg_watchdog() -> None:
    """Kill ffmpeg if no new chunks have hit disk for too long — the outer loop
    will respawn it. Avoids the 'process alive but frozen RTSP stream' case.

    The check is stateless (scans CHUNKS_DIR directly) so a processing backlog
    of older chunks can't suppress the real "ffmpeg producing now" signal.
    Previously the shared _last_chunk_mtime was updated by both ffmpeg output
    *and* backlog processing, which caused spurious kills at startup."""
    while True:
        time.sleep(30)
        try:
            chunk_files = list(CHUNKS_DIR.glob("chunk_*.wav"))
            if not chunk_files:
                # Nothing written yet — give ffmpeg a grace period from
                # whenever _last_chunk_mtime was last reset (at spawn).
                age = get_last_chunk_age()
            else:
                newest = max(c.stat().st_mtime for c in chunk_files)
                age = time.time() - newest
        except FileNotFoundError:
            continue
        except Exception:
            log.exception("watchdog scan failed")
            continue
        if age == 0.0:
            continue
        if age > FFMPEG_STALL_SECONDS:
            with _ffmpeg_pid_lock:
                pid = _ffmpeg_pid
            if pid is None:
                continue
            log.warning("ffmpeg watchdog: no new chunks for %.0fs (limit %ds); "
                        "killing pid %s so the loop restarts it", age,
                        FFMPEG_STALL_SECONDS, pid)
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(3)
                try:
                    os.kill(pid, 0)
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            except ProcessLookupError:
                pass
            except Exception:
                log.exception("failed killing ffmpeg")


def drop_excess_backlog() -> int:
    """If chunks are piling up faster than we can transcribe, drop the oldest."""
    chunks = sorted(CHUNKS_DIR.glob("chunk_*.wav"))
    if len(chunks) <= MAX_CHUNK_BACKLOG:
        return 0
    dropped = 0
    for c in chunks[:-MAX_CHUNK_BACKLOG]:
        try:
            c.unlink()
            dropped += 1
        except Exception:
            log.exception("failed to drop backlog chunk %s", c.name)
    if dropped:
        log.warning("chunk backlog exceeded %d; dropped %d oldest chunk(s)",
                    MAX_CHUNK_BACKLOG, dropped)
    return dropped


def uptime_kuma_heartbeat() -> None:
    """Optional: POST a heartbeat to an Uptime Kuma push monitor every minute
    when TRANSCRIBE_UPTIME_KUMA_PUSH_URL is set."""
    if not UPTIME_KUMA_PUSH_URL:
        return
    while True:
        status = "up"
        msg = "ok"
        try:
            age = get_last_chunk_age()
            if age > FFMPEG_STALL_SECONDS * 1.5:
                status = "down"
                msg = f"no chunks for {int(age)}s"
        except Exception:
            status = "down"
            msg = "heartbeat error"
        try:
            with httpx.Client(timeout=10.0) as c:
                c.get(UPTIME_KUMA_PUSH_URL, params={"status": status, "msg": msg})
        except Exception:
            log.debug("kuma push failed", exc_info=True)
        time.sleep(60)


def disk_space_guard() -> None:
    """Aggressively prune the audio archive when free space on /data dips below
    DISK_FREE_MIN_GB. Walks oldest days first until we're back above the line."""
    while True:
        try:
            usage = shutil.disk_usage(DATA_DIR)
            free_gb = usage.free / (1024 ** 3)
            if free_gb < DISK_FREE_MIN_GB:
                log.warning("Disk guard: free=%.1fGB < min=%.1fGB — pruning oldest audio",
                            free_gb, DISK_FREE_MIN_GB)
                for day_dir in sorted(AUDIO_DIR.glob("*")):
                    if not day_dir.is_dir():
                        continue
                    shutil.rmtree(day_dir, ignore_errors=True)
                    usage = shutil.disk_usage(DATA_DIR)
                    if usage.free / (1024 ** 3) >= DISK_FREE_MIN_GB:
                        break
        except Exception:
            log.exception("disk_space_guard pass failed")
        time.sleep(30 * 60)


def load_whisper() -> WhisperModel:
    log.info("Loading faster-whisper model=%s compute=%s threads=%d",
             MODEL_SIZE, COMPUTE_TYPE, CPU_THREADS)
    return WhisperModel(
        MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE,
        cpu_threads=CPU_THREADS, download_root=str(MODELS_DIR),
    )


def parse_chunk_time(path: pathlib.Path) -> dt.datetime:
    try:
        stem = path.stem.removeprefix("chunk_")
        return dt.datetime.strptime(stem, "%Y-%m-%dT%H-%M-%S")
    except ValueError:
        return dt.datetime.fromtimestamp(path.stat().st_mtime)


def archive_path_for(base: dt.datetime) -> pathlib.Path:
    return AUDIO_DIR / base.strftime("%Y-%m-%d") / base.strftime("%H-%M-%S.wav")


def archive_relative(base: dt.datetime) -> str:
    return f"{base.strftime('%Y-%m-%d')}/{base.strftime('%H-%M-%S.wav')}"


def process_chunk(
    model: WhisperModel, db: Database, embedder: Embedder | None,
    centroids_cache: dict, path: pathlib.Path,
) -> int:
    base = parse_chunk_time(path)
    try:
        touch_last_chunk(path.stat().st_mtime)
    except Exception:
        pass
    segments, info = model.transcribe(
        str(path),
        language=LANGUAGE,
        beam_size=BEAM_SIZE,
        initial_prompt=WHISPER_INITIAL_PROMPT or None,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": VAD_MIN_SILENCE_MS},
        condition_on_previous_text=False,
        word_timestamps=True,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        log_prob_threshold=LOG_PROB_THRESHOLD,
    )
    if LANGUAGE_PROB_MIN > 0 and getattr(info, "language_probability", 1.0) < LANGUAGE_PROB_MIN:
        log.info("%s: skipped (language_probability=%.2f < %.2f)",
                 path.name, info.language_probability, LANGUAGE_PROB_MIN)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return 0
    segments = list(segments)
    kept = []
    dropped = 0
    prev_text: str | None = None
    for s in segments:
        text = (s.text or "").strip()
        nsp = getattr(s, "no_speech_prob", None)
        avg_lp = getattr(s, "avg_logprob", None)
        cr = getattr(s, "compression_ratio", None)
        if is_hallucination(text, nsp, prev_text, avg_lp, cr):
            dropped += 1
            continue
        kept.append(s)
        prev_text = text
    if dropped:
        log.debug("%s: dropped %d hallucination/dup segment(s)", path.name, dropped)
    if not kept:
        return 0

    # LLM post-editing pass — rewrites mis-heard lines using rolling context.
    # Falls back to raw Whisper output on any failure.
    raw_texts = [(s.text or "").strip() for s in kept]
    corrected_texts = llm_correct_lines(raw_texts)
    _extend_recent_context(corrected_texts)

    # Load audio once for embedding extraction
    audio_array: np.ndarray | None = None
    sample_rate = 16000
    if embedder is not None:
        try:
            audio_array, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)
        except Exception:
            log.exception("Failed reading %s for embedding", path.name)
            audio_array = None

    # Archive chunk for later UI playback; write BEFORE DB rows reference it
    archive_dst = archive_path_for(base)
    archive_dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(path, archive_dst)
    except Exception:
        log.exception("Failed archiving %s", path.name)

    chunk_rel = archive_relative(base)
    log_lines: list[str] = []

    centroids = centroids_cache.get("list") or []
    for i, seg in enumerate(kept):
        text = corrected_texts[i] if i < len(corrected_texts) else (seg.text or "").strip()
        start_ts = base + dt.timedelta(seconds=float(seg.start))
        end_ts = base + dt.timedelta(seconds=float(seg.end))
        embedding: np.ndarray | None = None
        speaker_id: int | None = None
        if embedder is not None and audio_array is not None:
            s = max(int(seg.start * sample_rate), 0)
            e = min(int(seg.end * sample_rate), len(audio_array))
            if e > s:
                embedding = embedder.embed(audio_array[s:e])
        if embedding is not None:
            match = embedder.match(embedding, centroids, SPEAKER_MATCH_THRESHOLD) \
                if embedder is not None else None
            if match is not None:
                speaker_id = match.speaker_id
        seg_id = db.insert_segment(
            camera=CAMERA_NAME, start_ts=start_ts, end_ts=end_ts, text=text,
            speaker_id=speaker_id, chunk_file=chunk_rel,
            chunk_offset_s=float(seg.start), embedding=embedding,
        )
        log_lines.append(f"[{start_ts.strftime('%H:%M:%S')}] {text}")
        if KEYWORD_ALERTS:
            speaker_name = None
            if speaker_id is not None:
                for sid, name, _c in centroids:
                    if sid == speaker_id:
                        speaker_name = name
                        break
            fire_keyword_alerts(text, start_ts, seg_id, speaker_name)

    # Append to plain text daily log (human-readable fallback / backup)
    date_str = base.strftime("%Y-%m-%d")
    log_path = LOGS_DIR / f"{date_str}.log"
    try:
        with log_path.open("a") as fh:
            fh.write("\n".join(log_lines) + "\n")
    except Exception:
        log.exception("Failed writing log fallback")

    return len(kept)


def transcriber_loop(model: WhisperModel, db: Database, embedder: Embedder | None) -> None:
    centroids_cache: dict = {"list": [], "refreshed_at": 0.0}
    centroid_ttl = 120.0

    def refresh_centroids() -> None:
        try:
            centroids_cache["list"] = db.speaker_centroids() if embedder is not None else []
        except Exception:
            log.exception("Failed refreshing speaker centroids")
        centroids_cache["refreshed_at"] = time.monotonic()

    refresh_centroids()

    while True:
        if time.monotonic() - centroids_cache["refreshed_at"] > centroid_ttl:
            refresh_centroids()
        drop_excess_backlog()
        chunks = sorted(CHUNKS_DIR.glob("chunk_*.wav"))
        if chunks:
            try:
                touch_last_chunk(chunks[-1].stat().st_mtime)
            except FileNotFoundError:
                pass
        if len(chunks) < 2:
            time.sleep(3)
            continue
        for c in chunks[:-1]:
            try:
                count = process_chunk(model, db, embedder, centroids_cache, c)
                if count:
                    log.info("%s -> %d segment(s)", c.name, count)
            except Exception:
                log.exception("Processing failed for %s", c.name)
            finally:
                try:
                    c.unlink()
                except FileNotFoundError:
                    pass
        time.sleep(1)


def summary_scheduler(db: Database) -> None:
    """Midnight-ish daily summary of the completed previous day (with Pushover)."""
    hh, mm = (int(x) for x in SUMMARY_TIME.split(":"))
    while True:
        now = dt.datetime.now()
        target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if target <= now:
            target += dt.timedelta(days=1)
        delay = (target - now).total_seconds()
        log.info("Next end-of-day summary at %s local (sleep %.0fm)",
                 target.isoformat(), delay / 60)
        time.sleep(max(delay, 60))
        try:
            summariser.summarise_day(
                db, CAMERA_NAME, (dt.datetime.now() - dt.timedelta(days=1)).date(),
                ui_base_url=UI_BASE_URL or None, send_pushover=True, force=True,
            )
        except Exception:
            log.exception("Scheduled summary failed")


def rolling_summary_scheduler(db: Database) -> None:
    """At :05 past each hour, silently refresh today's summary so the UI stays
    current throughout the day. No Pushover — that's only at midnight."""
    while True:
        now = dt.datetime.now()
        target = now.replace(minute=5, second=0, microsecond=0)
        if target <= now:
            target += dt.timedelta(hours=1)
        delay = (target - now).total_seconds()
        log.info("Next rolling summary at %s local (sleep %.0fm)",
                 target.isoformat(), delay / 60)
        time.sleep(max(delay, 60))
        try:
            summariser.summarise_day(
                db, CAMERA_NAME, dt.date.today(),
                ui_base_url=UI_BASE_URL or None, send_pushover=False,
            )
        except Exception:
            log.exception("Rolling summary failed")


def audio_retention(db: Database) -> None:
    """Delete archived wav chunks older than AUDIO_RETENTION_DAYS. DB rows stay —
    only the audio files are pruned, so text transcripts are forever."""
    while True:
        try:
            cutoff = dt.date.today() - dt.timedelta(days=AUDIO_RETENTION_DAYS)
            for day_dir in sorted(AUDIO_DIR.glob("*")):
                if not day_dir.is_dir():
                    continue
                try:
                    day = dt.date.fromisoformat(day_dir.name)
                except ValueError:
                    continue
                if day < cutoff:
                    log.info("Pruning audio archive: %s", day_dir)
                    shutil.rmtree(day_dir, ignore_errors=True)
        except Exception:
            log.exception("audio_retention pass failed")
        # Run once a day
        time.sleep(24 * 3600)


def cleanup_stale_chunks() -> None:
    for c in CHUNKS_DIR.glob("chunk_*.wav"):
        try:
            c.unlink()
        except Exception:
            pass


def start_ui(db: Database, embedder: Embedder | None) -> None:
    app = ui_module.create_app(db, embedder)
    config = uvicorn.Config(app, host=UI_HOST, port=UI_PORT, log_level="info",
                            access_log=False, loop="asyncio")
    server = uvicorn.Server(config)
    server.run()


def main() -> None:
    os.environ["TZ"] = TZ_NAME
    try:
        time.tzset()
    except AttributeError:
        pass
    cleanup_stale_chunks()
    db = Database(DB_PATH)
    embedder = Embedder() if ENABLE_EMBEDDINGS else None
    whisper = load_whisper()

    threading.Thread(target=run_ffmpeg, name="ffmpeg", daemon=True).start()
    threading.Thread(target=ffmpeg_watchdog, name="ffmpeg-watchdog", daemon=True).start()
    threading.Thread(target=summary_scheduler, name="summary", args=(db,),
                     daemon=True).start()
    threading.Thread(target=rolling_summary_scheduler, name="rolling", args=(db,),
                     daemon=True).start()
    threading.Thread(target=audio_retention, name="retention", args=(db,),
                     daemon=True).start()
    threading.Thread(target=disk_space_guard, name="disk-guard", daemon=True).start()
    threading.Thread(target=uptime_kuma_heartbeat, name="kuma", daemon=True).start()
    threading.Thread(target=start_ui, name="ui", args=(db, embedder), daemon=True).start()

    transcriber_loop(whisper, db, embedder)


if __name__ == "__main__":
    main()
