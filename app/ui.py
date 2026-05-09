"""FastAPI UI for browsing transcripts, managing speakers, and reading summaries."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import logging
import os
import pathlib
import re
import sqlite3
import struct
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from db import Database
from embedder import Embedder, cluster_unknown_embeddings
import summariser


log = logging.getLogger("ui")


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n)
        if v not in (None, ""):
            return v
    return default


AUDIO_DIR = pathlib.Path(_env("AUDIO_DIR", default="/data/audio"))
CAMERA_NAME = _env("TRANSCRIBE_CAMERA", "CAMERA_NAME", default="dining")
SUMMARY_TIME = _env("TRANSCRIBE_SUMMARY_TIME", "SUMMARY_TIME", default="00:05")
UI_BASE_URL = _env("TRANSCRIBE_UI_BASE_URL", "UI_BASE_URL", default="").strip()
CONVERSATION_GAP_S = int(_env("TRANSCRIBE_CONVERSATION_GAP_S", default="180"))
SUGGEST_THRESHOLD = float(_env("TRANSCRIBE_SUGGEST_THRESHOLD", default="0.55"))
MENTION_WATCHLIST = [
    t.strip() for t in _env("TRANSCRIBE_MENTION_WATCHLIST", default="").split(",")
    if t.strip()
]

BASE_DIR = pathlib.Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

SPEAKER_PALETTE = [
    "#5aa9ff", "#ffd04b", "#ff8c61", "#4dd0a0", "#b982ff",
    "#ff6b9b", "#74d4ff", "#ffa062", "#9bd76a", "#ff5d7c",
    "#a3e0ff", "#ffc26b", "#d8a5ff",
]


# ---- helpers ---------------------------------------------------------------

def _fmt_ts(value: str) -> str:
    try:
        return value.replace("T", " ")
    except AttributeError:
        return str(value)


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _parse_ts(value: str) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(value)
    except Exception:
        return None


def _speaker_color(name: str | None) -> str:
    if not name or name.lower() == "unknown":
        return "#6b7280"
    digest = hashlib.md5(name.strip().lower().encode()).digest()[0]
    return SPEAKER_PALETTE[digest % len(SPEAKER_PALETTE)]


_TIMESTAMP_RE = re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?:\s*[-–]\s*([01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?)?\b")


def _linkify_summary(text: str, date_str: str) -> Markup:
    """Turn HH:MM / HH:MM-HH:MM substrings in a summary into jump-links into
    the day page at that time."""
    result: list[str] = []
    i = 0
    for m in _TIMESTAMP_RE.finditer(text):
        result.append(escape(text[i:m.start()]))
        token = m.group(0)
        # Use the first HH:MM of the match as the jump anchor
        first = re.match(r"([01]?\d|2[0-3]):[0-5]\d", token).group(0)
        hh, mm = first.split(":")
        anchor = f"/day/{date_str}#time-{int(hh):02d}-{int(mm):02d}"
        result.append(f'<a class="jump" href="{escape(anchor)}">{escape(token)}</a>')
        i = m.end()
    result.append(escape(text[i:]))
    return Markup("".join(result))


def _mention_patterns(terms: Iterable[str]) -> list[tuple[str, re.Pattern]]:
    """Return [(term, compiled regex)] — matches whole-word occurrences plus
    common possessive/plural trailing chars (e.g. Alex's, Friends)."""
    return [(t, re.compile(rf"\b{re.escape(t)}\w*\b", re.I)) for t in terms]


def _highlight_mentions(text: str, terms: Iterable[str] | None = None) -> Markup:
    """Wrap each watched-term occurrence in <mark class="mention">."""
    text = text or ""
    watchlist = list(terms) if terms is not None else MENTION_WATCHLIST
    if not watchlist or not text:
        return Markup(escape(text))
    combined = "|".join(rf"\b{re.escape(t)}\w*\b" for t in watchlist)
    pat = re.compile(combined, re.I)
    out: list[str] = []
    i = 0
    for m in pat.finditer(text):
        out.append(escape(text[i:m.start()]))
        out.append(f'<mark class="mention">{escape(m.group(0))}</mark>')
        i = m.end()
    out.append(escape(text[i:]))
    return Markup("".join(out))


def _extract_mentions(segments: list, terms: Iterable[str]) -> dict[str, list]:
    """Map term -> list of segment rows that mention it (in order)."""
    patterns = _mention_patterns(terms)
    buckets: dict[str, list] = {t: [] for t, _ in patterns}
    for s in segments:
        text = s["text"] or ""
        for term, pat in patterns:
            if pat.search(text):
                buckets[term].append(s)
    return buckets


templates.env.filters["fmt_ts"] = _fmt_ts
templates.env.filters["fmt_duration"] = _fmt_duration
templates.env.filters["speaker_color"] = _speaker_color
templates.env.filters["linkify_summary"] = _linkify_summary
templates.env.filters["highlight_mentions"] = _highlight_mentions
templates.env.globals["mention_terms"] = MENTION_WATCHLIST


@dataclass
class SpeakerStat:
    name: str
    color: str
    segment_count: int = 0
    speech_seconds: float = 0.0
    last_ts: dt.datetime | None = None


@dataclass
class ConversationBlock:
    start_ts: dt.datetime
    end_ts: dt.datetime
    segments: list = field(default_factory=list)
    speakers: set = field(default_factory=set)

    @property
    def duration_s(self) -> float:
        return (self.end_ts - self.start_ts).total_seconds()

    @property
    def segment_count(self) -> int:
        return len(self.segments)


def compute_day_stats(segments: list) -> dict:
    per_speaker: dict[str, SpeakerStat] = {}
    total_seconds = 0.0
    first_ts: dt.datetime | None = None
    last_ts: dt.datetime | None = None
    for s in segments:
        dur = float(s["duration_s"] or 0.0)
        total_seconds += dur
        label = s["speaker_name"] or "Unknown"
        stat = per_speaker.setdefault(label, SpeakerStat(name=label, color=_speaker_color(label)))
        stat.segment_count += 1
        stat.speech_seconds += dur
        start = _parse_ts(s["start_ts"])
        end = _parse_ts(s["end_ts"])
        if start is not None:
            first_ts = start if first_ts is None or start < first_ts else first_ts
        if end is not None:
            last_ts = end if last_ts is None or end > last_ts else last_ts
            if stat.last_ts is None or end > stat.last_ts:
                stat.last_ts = end
    ranked = sorted(per_speaker.values(),
                    key=lambda x: (-x.speech_seconds, x.name))
    max_speech = max((s.speech_seconds for s in ranked), default=0.0) or 1.0
    return {
        "segment_count": len(segments),
        "total_seconds": total_seconds,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "per_speaker": ranked,
        "max_speech": max_speech,
        "distinct_speakers": sum(1 for s in ranked if s.name != "Unknown"),
    }


def compute_hourly_timeline(segments: list) -> list[dict]:
    """Per-hour buckets (0..23) with stacked per-speaker speech seconds. Used for
    the heat-bar visualisation."""
    buckets = [
        {"hour": h, "total_s": 0.0, "by_speaker": {}} for h in range(24)
    ]
    for s in segments:
        start = _parse_ts(s["start_ts"])
        end = _parse_ts(s["end_ts"])
        if start is None or end is None:
            continue
        label = s["speaker_name"] or "Unknown"
        cur = start
        while cur < end:
            nxt_hour = (cur.replace(minute=0, second=0, microsecond=0)
                        + dt.timedelta(hours=1))
            slice_end = min(end, nxt_hour)
            secs = (slice_end - cur).total_seconds()
            b = buckets[cur.hour]
            b["total_s"] += secs
            b["by_speaker"][label] = b["by_speaker"].get(label, 0.0) + secs
            cur = slice_end
    max_total = max((b["total_s"] for b in buckets), default=0.0) or 1.0
    for b in buckets:
        b["height_pct"] = round(min(b["total_s"] / max_total, 1.0) * 100, 1)
        b["stacks"] = [
            {
                "name": name,
                "color": _speaker_color(name),
                "pct_of_hour": (secs / b["total_s"] * 100) if b["total_s"] else 0,
            }
            for name, secs in sorted(b["by_speaker"].items(),
                                     key=lambda kv: -kv[1])
        ]
    return buckets


def group_conversations(segments: list, gap_s: int = CONVERSATION_GAP_S) -> list[ConversationBlock]:
    blocks: list[ConversationBlock] = []
    current: ConversationBlock | None = None
    for s in segments:
        start = _parse_ts(s["start_ts"])
        end = _parse_ts(s["end_ts"])
        if start is None or end is None:
            continue
        label = s["speaker_name"] or "Unknown"
        if current is None or (start - current.end_ts).total_seconds() > gap_s:
            current = ConversationBlock(start_ts=start, end_ts=end)
            blocks.append(current)
        current.segments.append(s)
        current.speakers.add(label)
        if end > current.end_ts:
            current.end_ts = end
    return blocks


def _suggestions_for_segments(segments: list, embedder: Embedder | None,
                              centroids: list) -> dict[int, dict]:
    """Map segment_id -> suggestion dict for unlabeled segments whose embedding
    is close enough to a named speaker's centroid to bother recommending."""
    if embedder is None or not centroids:
        return {}
    out: dict[int, dict] = {}
    for s in segments:
        if s["speaker_id"] is not None:
            continue
        emb_bytes = s["embedding"]
        if not emb_bytes:
            continue
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        best = embedder.best_match(emb, centroids)
        if best is None or best.similarity < SUGGEST_THRESHOLD:
            continue
        out[int(s["id"])] = {
            "speaker_id": best.speaker_id,
            "name": best.name,
            "similarity": best.similarity,
            "color": _speaker_color(best.name),
        }
    return out


# ---- app factory ------------------------------------------------------------

def create_app(db: Database, embedder: Embedder | None) -> FastAPI:
    app = FastAPI(title=f"Frigate transcriber — {CAMERA_NAME}")

    static_dir = BASE_DIR / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def render_segment_row(request: Request, seg, speakers, *,
                           suggestion: dict | None = None,
                           edit: bool = False) -> HTMLResponse:
        template = "_segment_row_edit.html" if edit else "_segment_row.html"
        return templates.TemplateResponse(template, {
            "request": request,
            "seg": seg,
            "speakers": speakers,
            "suggestion": suggestion,
        })

    # ---- pages --------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index():
        today = dt.date.today().isoformat()
        return RedirectResponse(f"/day/{today}", status_code=302)

    @app.get("/day/{date}", response_class=HTMLResponse)
    def day(request: Request, date: str,
            hide_unknown: int = Query(0),
            speaker_id: int | None = Query(None)):
        try:
            dt.date.fromisoformat(date)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        all_segments = db.segments_for_date(date, camera=CAMERA_NAME)
        stats = compute_day_stats(all_segments)
        timeline = compute_hourly_timeline(all_segments)

        filtered = all_segments
        if hide_unknown:
            filtered = [s for s in filtered if s["speaker_id"]]
        if speaker_id is not None:
            filtered = [s for s in filtered if s["speaker_id"] == speaker_id]

        blocks = group_conversations(filtered)
        speakers = db.list_speakers()
        dates = db.dates_with_segments()
        summary_row = db.summary(CAMERA_NAME, date)
        today_str = dt.date.today().isoformat()

        centroids = db.speaker_centroids() if embedder is not None else []
        suggestions = _suggestions_for_segments(filtered, embedder, centroids)

        unknown_count = sum(1 for s in all_segments if not s["speaker_id"])
        mentions = _extract_mentions(all_segments, MENTION_WATCHLIST)
        mention_counts = {t: len(v) for t, v in mentions.items()}
        speaker_name_to_id = {sp["name"]: sp["id"] for sp in speakers}
        return templates.TemplateResponse("day.html", {
            "request": request,
            "date": date,
            "is_today": date == today_str,
            "segments": filtered,
            "all_segments": all_segments,
            "unknown_count": unknown_count,
            "blocks": blocks,
            "stats": stats,
            "timeline": timeline,
            "suggestions": suggestions,
            "mentions": mentions,
            "mention_counts": mention_counts,
            "watchlist": MENTION_WATCHLIST,
            "speaker_name_to_id": speaker_name_to_id,
            "now": dt.datetime.now(),
            "hide_unknown": bool(hide_unknown),
            "filter_speaker_id": speaker_id,
            "speakers": speakers,
            "dates": dates,
            "camera": CAMERA_NAME,
            "summary_row": summary_row,
            "summary_time": SUMMARY_TIME,
            "prev_date": (dt.date.fromisoformat(date) - dt.timedelta(days=1)).isoformat(),
            "next_date": (dt.date.fromisoformat(date) + dt.timedelta(days=1)).isoformat(),
            "today": today_str,
        })

    @app.get("/search", response_class=HTMLResponse)
    def search(request: Request,
               q: str = Query("", max_length=200),
               date_from: str | None = Query(None),
               date_to: str | None = Query(None),
               speaker_id: int | None = Query(None),
               limit: int = Query(200, ge=1, le=2000)):
        results: list[sqlite3.Row] = []
        if q:
            results = db.search(q, limit=limit, date_from=date_from or None,
                                date_to=date_to or None, speaker_id=speaker_id)
        speakers = db.list_speakers()
        return templates.TemplateResponse("search.html", {
            "request": request,
            "q": q,
            "results": results,
            "speakers": speakers,
            "date_from": date_from or "",
            "date_to": date_to or "",
            "speaker_id": speaker_id,
            "camera": CAMERA_NAME,
        })

    @app.get("/speakers", response_class=HTMLResponse)
    def speakers_page(request: Request):
        speakers = db.list_speakers()
        return templates.TemplateResponse("speakers.html", {
            "request": request, "speakers": speakers, "camera": CAMERA_NAME,
        })

    @app.get("/speakers/{speaker_id}", response_class=HTMLResponse)
    def speaker_page(request: Request, speaker_id: int):
        row = db.conn.execute(
            "SELECT * FROM speakers WHERE id=?", (speaker_id,),
        ).fetchone()
        if not row:
            raise HTTPException(404)
        segments = db.segments_for_speaker(speaker_id, limit=500)
        other_speakers = [s for s in db.list_speakers() if s["id"] != speaker_id]
        stats = compute_day_stats(segments)
        return templates.TemplateResponse("speaker.html", {
            "request": request, "speaker": row, "segments": segments,
            "stats": stats, "camera": CAMERA_NAME, "other_speakers": other_speakers,
        })

    @app.get("/summaries", response_class=HTMLResponse)
    def summaries_page(request: Request):
        return templates.TemplateResponse("summaries.html", {
            "request": request, "summaries": db.recent_summaries(),
            "camera": CAMERA_NAME,
        })

    @app.get("/summaries/{date}", response_class=HTMLResponse)
    def summary_page(request: Request, date: str):
        row = db.summary(CAMERA_NAME, date)
        if not row:
            raise HTTPException(404, "no summary for that date")
        return templates.TemplateResponse("summary.html", {
            "request": request, "summary": row, "date": date, "camera": CAMERA_NAME,
        })

    @app.get("/mentions", response_class=HTMLResponse)
    def mentions_page(request: Request,
                      term: str | None = Query(None),
                      date_from: str | None = Query(None),
                      date_to: str | None = Query(None),
                      limit: int = Query(500, ge=1, le=5000)):
        terms = [t for t in MENTION_WATCHLIST if (not term or t.lower() == term.lower())]
        per_term: dict[str, list] = {}
        for t in terms:
            pattern = rf"\b{re.escape(t)}\w*\b"
            sql = ["text REGEXP ? "]
            args: list = [pattern]
            if date_from:
                sql.append("AND substr(start_ts, 1, 10) >= ?")
                args.append(date_from)
            if date_to:
                sql.append("AND substr(start_ts, 1, 10) <= ?")
                args.append(date_to)
            args.append(limit)
            # SQLite lacks REGEXP by default; use LIKE + Python filter to stay simple.
            like_rows = db.search(t, limit=limit * 3, date_from=date_from,
                                  date_to=date_to)
            pat = re.compile(pattern, re.I)
            matched = [r for r in like_rows if pat.search(r["text"] or "")]
            per_term[t] = matched[:limit]
        return templates.TemplateResponse("mentions.html", {
            "request": request, "watchlist": MENTION_WATCHLIST,
            "per_term": per_term, "camera": CAMERA_NAME,
            "date_from": date_from or "", "date_to": date_to or "",
            "active_term": term or "",
        })

    @app.get("/clusters", response_class=HTMLResponse)
    def clusters_page(request: Request,
                      min_size: int = Query(3, ge=2, le=50),
                      threshold: float = Query(0.78, ge=0.5, le=0.95)):
        rows = db.segments_with_embeddings(speaker_id=None, limit=20000)
        items = [(int(r["id"]), np.frombuffer(r["embedding"], dtype=np.float32))
                 for r in rows]
        clusters = cluster_unknown_embeddings(items, threshold=threshold, min_size=min_size)
        centroids = db.speaker_centroids() if embedder is not None else []
        enriched = []
        samples: dict[int, list] = {}
        if clusters:
            ids = [sid for c in clusters for sid in c.sample_segment_ids]
            if ids:
                placeholders = ",".join("?" * len(ids))
                seg_rows = db.conn.execute(f"""
                    SELECT seg.id, seg.start_ts, seg.text, seg.chunk_file,
                           seg.chunk_offset_s, seg.duration_s
                      FROM segments seg WHERE seg.id IN ({placeholders})
                """, ids).fetchall()
                seg_by_id = {r["id"]: r for r in seg_rows}
                for c in clusters:
                    samples[c.cluster_id] = [seg_by_id[i] for i in c.sample_segment_ids
                                             if i in seg_by_id]
        for c in clusters:
            best = (embedder.best_match(c.centroid, centroids)
                    if embedder is not None and centroids else None)
            enriched.append({
                "cluster_id": c.cluster_id,
                "size": c.size,
                "member_ids": c.member_ids,
                "member_ids_json": json.dumps(c.member_ids),
                "samples": samples.get(c.cluster_id, []),
                "best": {
                    "name": best.name, "similarity": best.similarity,
                    "speaker_id": best.speaker_id, "color": _speaker_color(best.name),
                } if best is not None and best.similarity >= 0.45 else None,
            })
        return templates.TemplateResponse("clusters.html", {
            "request": request, "clusters": enriched,
            "speakers": db.list_speakers(), "camera": CAMERA_NAME,
            "threshold": threshold, "min_size": min_size,
        })

    # ---- mutations ----------------------------------------------------------

    @app.post("/api/speakers")
    def api_create_speaker(name: str = Form(...), notes: str | None = Form(None)):
        sid = db.ensure_speaker(name, notes)
        return RedirectResponse("/speakers", status_code=303)

    @app.post("/api/speakers/{speaker_id}/rename")
    def api_rename_speaker(speaker_id: int, name: str = Form(...)):
        db.rename_speaker(speaker_id, name)
        return RedirectResponse("/speakers", status_code=303)

    @app.post("/api/speakers/{speaker_id}/merge")
    def api_merge_speaker(speaker_id: int, into_id: int = Form(...)):
        moved = db.merge_speakers(speaker_id, into_id)
        log.info("Merged speaker %d into %d (%d segments moved)", speaker_id, into_id, moved)
        return RedirectResponse(f"/speakers/{into_id}", status_code=303)

    @app.post("/api/speakers/{speaker_id}/delete")
    def api_delete_speaker(speaker_id: int):
        db.delete_speaker(speaker_id)
        return RedirectResponse("/speakers", status_code=303)

    @app.post("/api/segments/{segment_id}/speaker", response_class=HTMLResponse)
    def api_set_segment_speaker(request: Request, segment_id: int,
                                speaker_id: str = Form(""), new_name: str = Form(""),
                                propagate: str = Form("")):
        seg = db.segment(segment_id)
        if not seg:
            raise HTTPException(404)
        if speaker_id == "__none__":
            target_id = None
        elif speaker_id == "__new__" and new_name.strip():
            target_id = db.ensure_speaker(new_name.strip())
        else:
            try:
                target_id = int(speaker_id) if speaker_id else None
            except ValueError:
                raise HTTPException(400, "bad speaker_id")
        db.set_segment_speaker(segment_id, target_id, locked=True)

        if propagate and target_id is not None and embedder is not None \
                and seg["embedding"] is not None:
            centroids = db.speaker_centroids()
            centroid_map = {sid: c for sid, _n, c in centroids}
            target_centroid = centroid_map.get(target_id)
            if target_centroid is not None:
                unlabeled = db.segments_with_embeddings(speaker_id=None, limit=10000)
                similar_ids: list[int] = []
                # 0.85 — distant ceiling-mic embeddings drift, so propagate only
                # on strong matches; 0.78 was producing a lot of false positives.
                for u in unlabeled:
                    emb = np.frombuffer(u["embedding"], dtype=np.float32)
                    nrm = float(np.linalg.norm(emb)) or 1.0
                    sim = float(np.dot(emb / nrm, target_centroid))
                    if sim >= 0.85:
                        similar_ids.append(int(u["id"]))
                db.propagate_speaker(target_id, similar_ids)

        updated = db.segment(segment_id)
        return render_segment_row(request, updated, db.list_speakers())

    @app.get("/api/segments/{segment_id}/edit", response_class=HTMLResponse)
    def api_segment_edit_form(request: Request, segment_id: int):
        seg = db.segment(segment_id)
        if not seg:
            raise HTTPException(404)
        return render_segment_row(request, seg, db.list_speakers(), edit=True)

    @app.get("/api/segments/{segment_id}/view", response_class=HTMLResponse)
    def api_segment_view(request: Request, segment_id: int):
        seg = db.segment(segment_id)
        if not seg:
            raise HTTPException(404)
        return render_segment_row(request, seg, db.list_speakers())

    @app.post("/api/segments/{segment_id}/text", response_class=HTMLResponse)
    def api_segment_update_text(request: Request, segment_id: int,
                                text: str = Form(...)):
        try:
            db.update_segment_text(segment_id, text)
        except ValueError as e:
            raise HTTPException(400, str(e))
        seg = db.segment(segment_id)
        return render_segment_row(request, seg, db.list_speakers())

    @app.post("/api/segments/{segment_id}/delete")
    def api_segment_delete(segment_id: int):
        db.delete_segment(segment_id)
        return PlainTextResponse("", status_code=200)

    @app.post("/api/clusters/label")
    def api_cluster_label(speaker_id: str = Form(""),
                          new_name: str = Form(""),
                          segment_ids: str = Form(...)):
        try:
            ids = json.loads(segment_ids)
        except Exception as e:
            raise HTTPException(400, f"bad segment_ids JSON: {e}")
        if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
            raise HTTPException(400, "segment_ids must be a JSON array of ints")
        if speaker_id == "__new__" and new_name.strip():
            target_id = db.ensure_speaker(new_name.strip())
        else:
            try:
                target_id = int(speaker_id)
            except ValueError:
                raise HTTPException(400, "bad speaker_id")
        with db.tx() as c:
            placeholders = ",".join("?" * len(ids))
            c.execute(
                f"UPDATE segments SET speaker_id=?, speaker_locked=1 "
                f" WHERE id IN ({placeholders})",
                (target_id, *ids),
            )
        return RedirectResponse("/clusters", status_code=303)

    @app.post("/api/summary/now")
    def api_summary_now(date: str = Form(...)):
        try:
            target = dt.date.fromisoformat(date)
        except ValueError as e:
            raise HTTPException(400, detail=str(e))
        summariser.summarise_day(db, CAMERA_NAME, target,
                                 ui_base_url=UI_BASE_URL or None,
                                 send_pushover=False)
        return RedirectResponse(f"/day/{date}", status_code=303)

    # ---- audio playback -----------------------------------------------------

    @app.get("/api/segments/{segment_id}/audio")
    def api_segment_audio(segment_id: int):
        seg = db.segment(segment_id)
        if not seg:
            raise HTTPException(404)
        chunk_file = seg["chunk_file"]
        offset = float(seg["chunk_offset_s"] or 0.0)
        duration = float(seg["duration_s"] or 0.0) + 0.4
        if not chunk_file:
            raise HTTPException(404, "no audio file for this segment")
        path = AUDIO_DIR / chunk_file
        if not path.is_file():
            raise HTTPException(404, "audio file missing")
        try:
            data, sr = sf.read(str(path), dtype="int16")
        except Exception as e:
            raise HTTPException(500, f"could not read audio: {e}")
        start_sample = max(int(offset * sr), 0)
        end_sample = min(int((offset + duration) * sr), len(data))
        snippet = data[start_sample:end_sample]
        channels = 1 if snippet.ndim == 1 else snippet.shape[1]
        buf = io.BytesIO()
        _write_wav(buf, snippet, sr, channels)
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/wav", headers={
            "Cache-Control": "public, max-age=604800",
        })

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "camera": CAMERA_NAME}

    return app


def _write_wav(out: io.BytesIO, samples: np.ndarray, sr: int, channels: int) -> None:
    samples = np.ascontiguousarray(samples, dtype=np.int16)
    byte_rate = sr * channels * 2
    block_align = channels * 2
    data_bytes = samples.tobytes()
    out.write(b"RIFF")
    out.write(struct.pack("<I", 36 + len(data_bytes)))
    out.write(b"WAVE")
    out.write(b"fmt ")
    out.write(struct.pack("<IHHIIHH", 16, 1, channels, sr, byte_rate, block_align, 16))
    out.write(b"data")
    out.write(struct.pack("<I", len(data_bytes)))
    out.write(data_bytes)
