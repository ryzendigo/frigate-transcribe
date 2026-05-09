"""Daily end-of-day summarisation, speaker-aware, pushed via Pushover."""

from __future__ import annotations

import datetime as dt
import logging
import os
import re
from typing import Iterable, Sequence

import httpx

from db import Database


log = logging.getLogger("summariser")


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n)
        if v not in (None, ""):
            return v
    return default


LLM_BACKEND = _env("TRANSCRIBE_LLM_BACKEND", "LLM_BACKEND", default="gemini").lower()
OLLAMA_URL = _env("OLLAMA_URL", default="http://ollama:11434").rstrip("/")
OLLAMA_MODEL = _env("TRANSCRIBE_OLLAMA_MODEL", "OLLAMA_MODEL", default="qwen3:4b")
GEMINI_API_KEY = _env("TRANSCRIBE_GEMINI_API_KEY", "GEMINI_API_KEY", default="")
GEMINI_MODEL = _env("TRANSCRIBE_GEMINI_MODEL", "GEMINI_MODEL",
                    default="gemini-2.5-flash-lite")
PUSHOVER_USER = _env("PUSHOVER_USER_KEY", default="")
PUSHOVER_TOKEN = _env("TRANSCRIBE_PUSHOVER_TOKEN", "PUSHOVER_TOKEN", default="")
MAX_PROMPT_CHARS = int(_env("SUMMARY_MAX_PROMPT_CHARS", default="140000"))
WATCHLIST = [
    t.strip() for t in _env("TRANSCRIBE_MENTION_WATCHLIST", default="").split(",")
    if t.strip()
]
SPEAKER_NAMES = [
    n.strip() for n in _env("TRANSCRIBE_SPEAKER_NAMES", default="").split(",")
    if n.strip()
]
MIN_SEGMENTS_FOR_SUMMARY = int(_env("TRANSCRIBE_MIN_SEGMENTS_FOR_SUMMARY",
                                    default="8"))
SUMMARY_MIN_INTERVAL_S = int(_env("TRANSCRIBE_SUMMARY_MIN_INTERVAL_S",
                                  default="30"))
_summary_throttle: dict[str, float] = {}
_summary_throttle_lock = __import__("threading").Lock()


def _strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _speaker_label(row) -> str:
    name = row["speaker_name"]
    if name:
        return name
    return "Unknown"


def _transcript_lines(segments: Sequence, *, include_speakers: bool) -> list[str]:
    lines: list[str] = []
    for s in segments:
        hhmmss = s["start_ts"][11:19] if len(s["start_ts"]) >= 19 else s["start_ts"]
        text = (s["text"] or "").strip()
        if not text:
            continue
        if include_speakers:
            lines.append(f"[{hhmmss}] {_speaker_label(s)}: {text}")
        else:
            lines.append(f"[{hhmmss}] {text}")
    return lines


def _count_by_speaker(segments: Sequence) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in segments:
        label = _speaker_label(s)
        counts[label] = counts.get(label, 0) + 1
    return counts


def _call_ollama(prompt: str) -> str:
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "think": False,
        "options": {"temperature": 0.3, "num_ctx": 8192},
    }
    with httpx.Client(timeout=3600.0) as c:
        r = c.post(url, json=payload)
        r.raise_for_status()
        return (r.json().get("response") or "").strip()


def _call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048},
    }
    with httpx.Client(timeout=180.0) as c:
        r = c.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Gemini response: {data}") from e


def _llm_generate(prompt: str) -> str:
    if LLM_BACKEND == "gemini":
        return _call_gemini(prompt)
    if LLM_BACKEND == "ollama":
        return _call_ollama(prompt)
    raise RuntimeError(f"Unknown LLM_BACKEND={LLM_BACKEND!r}")


def pushover_notify(title: str, message: str, *, priority: int = 0,
                    url: str | None = None, url_title: str | None = None) -> None:
    if not (PUSHOVER_USER and PUSHOVER_TOKEN):
        log.warning("Pushover not configured; skipping notification")
        return
    body = {
        "token": PUSHOVER_TOKEN, "user": PUSHOVER_USER,
        "title": title[:250], "message": message[:1024], "priority": str(priority),
    }
    if url:
        body["url"] = url[:512]
    if url_title:
        body["url_title"] = url_title[:100]
    try:
        with httpx.Client(timeout=30.0) as c:
            c.post("https://api.pushover.net/1/messages.json", data=body).raise_for_status()
    except Exception:
        log.exception("Pushover notify failed")


def build_prompt(*, camera: str, target_date: dt.date, lines: Iterable[str],
                 speaker_counts: dict[str, int],
                 speakers_in_lines: bool) -> str:
    speaker_line = ", ".join(
        f"{name} ({count} line(s))" for name, count in sorted(
            speaker_counts.items(), key=lambda kv: (-kv[1], kv[0]),
        )
    ) or "(none)"
    watchlist = ", ".join(WATCHLIST) if WATCHLIST else "(none)"
    transcript = "\n".join(lines)
    if len(transcript) > MAX_PROMPT_CHARS:
        transcript = transcript[-MAX_PROMPT_CHARS:]
        transcript = "(... earlier lines truncated to fit context ...)\n" + transcript

    mention_sections = ""
    for term in WATCHLIST:
        mention_sections += (
            f"MENTIONS of {term.upper()}:\n"
            f"  List every transcript line where '{term}' (or related forms like "
            f"{term}'s, {term}s, addressing {term} directly) appears in the TEXT of "
            f"the line. Use the exact quote from the transcript. Format each one as:\n"
            f"  - HH:MM: \"quote that includes {term}\" — brief context.\n"
            f"  If nothing was said: none.\n\n"
        )

    if speakers_in_lines:
        named_in_transcript = [n for n in speaker_counts.keys() if n != "Unknown"]
        format_rule = (
            "Each transcript line looks like '[HH:MM:SS] NAME: text' where NAME is "
            "either a specific attributed person or 'Unknown' for unattributed speech. "
            "Only attribute a quote to a named person if that EXACT name appears as "
            "the speaker label in the line. The only named speakers actually "
            f"attributed in this day's transcript are: "
            f"{', '.join(named_in_transcript) or '(none)'}. If a line starts with "
            "'Unknown:' you MUST refer to that speaker as 'someone' or 'an unknown "
            "speaker'."
        )
    else:
        names_warning = (
            f" (e.g. {', '.join(SPEAKER_NAMES)})" if SPEAKER_NAMES else ""
        )
        format_rule = (
            "Each transcript line looks like '[HH:MM:SS] text' — there is NO speaker "
            "label on any line because no voices have been attributed yet. Therefore "
            "you MUST refer to every speaker as 'someone' or 'an unknown speaker'. "
            f"NEVER invent a speaker name{names_warning}. Even if the content "
            "strongly suggests who is speaking, you have no way to know — do not "
            "guess."
        )

    return (
        f"You are reading a timestamped speech-to-text transcript captured near the "
        f"'{camera}' camera on {target_date.isoformat()}. Expect artefacts: misheard "
        f"words, overlapping speech, TV/music, ambient noise.\n\n"
        f"SPEAKER BREAKDOWN FOR THIS DAY: {speaker_line}\n"
        f"WATCHLIST (names the reader especially cares about hearing mentioned in the "
        f"text — these are PEOPLE NAMED by speakers, not speakers themselves): "
        f"{watchlist}\n\n"
        "CRITICAL RULES (non-negotiable):\n"
        f"1. {format_rule}\n"
        "2. Never invent content. If something is not in the transcript, do not write it.\n"
        "3. Never assume two unattributed lines are the same person unless the text "
        "   content itself proves it (e.g. self-introduction).\n"
        "4. The WATCHLIST is about names APPEARING IN TEXT, not speaker attributions.\n\n"
        "Write a detailed, specific, plain-text daily summary. NO markdown headings, "
        "NO preamble, NO meta-commentary. Extract real content: names dropped in "
        "conversation, numbers, places, plans, decisions, and short direct quotes. "
        "Avoid vague wording like 'discussed a topic' — say what the topic was.\n\n"
        "Respond in EXACTLY this structure, with these uppercase section labels in "
        "this order, each on its own line. Use plain '- ' bullets.\n\n"
        "TL;DR: one sentence under 180 characters capturing the shape of the day.\n\n"
        f"{mention_sections}"
        "HIGHLIGHTS:\n"
        "  3–7 bullets of the most noteworthy content: decisions, plans, arguments, "
        "news, or standout things that were said. Each bullet format: "
        "'- HH:MM <speaker>: short direct quote from the transcript — context'. Use "
        "'someone' when no speaker name is attributed in the transcript.\n\n"
        "PER-SPEAKER RECAP:\n"
        "  One bullet PER named speaker actually attributed in the transcript "
        "(skip 'Unknown'). Each bullet: '- <Name>: 1-2 sentences on what THIS "
        "person focused on today — topics, questions, opinions, feelings — with "
        "a characteristic direct quote if available'. If no named speakers are "
        "attributed yet, write: none.\n\n"
        "VISITORS / NEW VOICES:\n"
        "  Note if a distinctly different voice dominated any long stretch (possible "
        "visitor). Otherwise: none.\n\n"
        "TIMELINE:\n"
        "  6–14 bullets walking the whole active day in order. Each: "
        "'- HH:MM-HH:MM: concrete topic with at least one specific detail (a name, "
        "a number, a decision, or a short direct quote)'. Group related chatter. Skip "
        "stretches dominated by TV/music/ambient rather than real conversation.\n\n"
        "Keep the total under 2600 characters. Be blunt, factual, and specific.\n\n"
        f"Transcript:\n{transcript}"
    )


def _should_throttle(key: str) -> bool:
    import time as _t
    with _summary_throttle_lock:
        now = _t.monotonic()
        last = _summary_throttle.get(key, 0.0)
        if now - last < SUMMARY_MIN_INTERVAL_S:
            return True
        _summary_throttle[key] = now
        return False


def summarise_day(db: Database, camera: str, target_date: dt.date, *,
                  ui_base_url: str | None = None,
                  send_pushover: bool = True,
                  force: bool = False) -> str | None:
    date_str = target_date.isoformat()
    key = f"{camera}/{date_str}"
    if not force and _should_throttle(key):
        log.info("Summary throttled for %s (min interval %ds)", key,
                 SUMMARY_MIN_INTERVAL_S)
        return None
    segments = db.segments_for_date(date_str, camera=camera)
    if not segments:
        log.info("No segments for %s / %s; skipping summary", camera, date_str)
        return None
    if not force and len(segments) < MIN_SEGMENTS_FOR_SUMMARY:
        log.info("Only %d segments for %s; below %d threshold, skipping",
                 len(segments), date_str, MIN_SEGMENTS_FOR_SUMMARY)
        return None
    counts = _count_by_speaker(segments)
    named_count = sum(n for name, n in counts.items() if name != "Unknown")
    include_speakers = named_count > 0
    lines = _transcript_lines(segments, include_speakers=include_speakers)
    if not lines:
        log.info("Segments exist for %s but all empty; skipping", date_str)
        return None
    prompt = build_prompt(
        camera=camera, target_date=target_date, lines=lines, speaker_counts=counts,
        speakers_in_lines=include_speakers,
    )
    backend_desc = (
        f"{LLM_BACKEND}/{GEMINI_MODEL if LLM_BACKEND == 'gemini' else OLLAMA_MODEL}"
    )
    log.info("Summarising %s / %s (%d segments, %d chars) via %s",
             camera, date_str, len(segments), len(prompt), backend_desc)
    try:
        raw = _llm_generate(prompt)
    except Exception as e:
        log.exception("LLM summarise failed (%s)", backend_desc)
        if send_pushover:
            pushover_notify(
                f"{camera} summary FAILED {date_str}", f"LLM error: {e}", priority=-1,
            )
        return None
    summary = _strip_think_blocks(raw)
    db.save_summary(
        camera=camera, summary_date=date_str, segment_count=len(segments), summary=summary,
    )
    if send_pushover:
        url = None
        if ui_base_url:
            url = f"{ui_base_url.rstrip('/')}/summaries/{date_str}"
        pushover_notify(
            f"{camera} transcript — {date_str}",
            summary,
            url=url,
            url_title="Open full transcript" if url else None,
        )
    return summary
