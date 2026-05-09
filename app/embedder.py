"""Speaker embeddings via Resemblyzer + nearest-centroid attribution."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from resemblyzer import VoiceEncoder


log = logging.getLogger("embedder")


@dataclass(frozen=True)
class SpeakerMatch:
    speaker_id: int
    name: str
    similarity: float


class Embedder:
    """Wraps a resemblyzer VoiceEncoder. Safe to call from a single thread."""

    def __init__(self) -> None:
        log.info("Loading Resemblyzer voice encoder (CPU)")
        self.encoder = VoiceEncoder(device="cpu", verbose=False)
        self.sample_rate = 16000  # resemblyzer expects 16kHz float waveforms

    def embed(self, wav: np.ndarray) -> np.ndarray | None:
        """Return a unit-norm 256-d float32 embedding, or None if audio too short."""
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = np.asarray(wav, dtype=np.float32)
        # Resemblyzer needs at least ~1.6s of audio for a partial window; for anything
        # shorter we pad with silence rather than skip — this keeps short utterances
        # ("yes", "okay") addressable even if the embedding is less discriminative.
        min_samples = int(1.6 * self.sample_rate)
        if len(wav) < min_samples:
            pad = np.zeros(min_samples - len(wav), dtype=np.float32)
            wav = np.concatenate([pad, wav])
        try:
            emb = self.encoder.embed_utterance(wav)
        except Exception as e:
            log.warning("Resemblyzer failed on %.2fs audio: %s", len(wav) / self.sample_rate, e)
            return None
        emb = np.asarray(emb, dtype=np.float32)
        norm = float(np.linalg.norm(emb)) or 1.0
        return emb / norm

    @staticmethod
    def match(
        embedding: np.ndarray,
        centroids: Sequence[tuple[int, str, np.ndarray]],
        threshold: float,
    ) -> SpeakerMatch | None:
        best = Embedder.best_match(embedding, centroids)
        if best is None or best.similarity < threshold:
            return None
        return best

    @staticmethod
    def best_match(
        embedding: np.ndarray,
        centroids: Sequence[tuple[int, str, np.ndarray]],
    ) -> SpeakerMatch | None:
        if not centroids or embedding is None:
            return None
        norm = float(np.linalg.norm(embedding)) or 1.0
        emb = embedding / norm
        best: SpeakerMatch | None = None
        for sid, name, centroid in centroids:
            sim = float(np.dot(emb, centroid))
            if best is None or sim > best.similarity:
                best = SpeakerMatch(sid, name, sim)
        return best


@dataclass
class UnknownCluster:
    cluster_id: int
    size: int
    member_ids: list[int]
    centroid: np.ndarray
    best_named_match: SpeakerMatch | None
    sample_segment_ids: list[int]


def cluster_unknown_embeddings(
    items: Sequence[tuple[int, np.ndarray]],
    threshold: float = 0.78,
    min_size: int = 3,
    max_clusters: int = 30,
) -> list[UnknownCluster]:
    """Greedy agglomerative clustering over unit-norm embeddings.

    Each item = (segment_id, raw_embedding). Returns clusters of at least
    `min_size` members, largest first.
    """
    if not items:
        return []
    normed: list[tuple[int, np.ndarray]] = []
    for sid, emb in items:
        if emb is None:
            continue
        n = float(np.linalg.norm(emb)) or 1.0
        normed.append((int(sid), np.asarray(emb, dtype=np.float32) / n))

    centroids: list[np.ndarray] = []
    members: list[list[int]] = []
    counts: list[int] = []
    for sid, emb in normed:
        best_idx = -1
        best_sim = -1.0
        for i, c in enumerate(centroids):
            sim = float(np.dot(emb, c))
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        if best_idx >= 0 and best_sim >= threshold:
            n_old = counts[best_idx]
            centroids[best_idx] = (centroids[best_idx] * n_old + emb) / (n_old + 1)
            centroids[best_idx] /= float(np.linalg.norm(centroids[best_idx])) or 1.0
            counts[best_idx] = n_old + 1
            members[best_idx].append(sid)
        else:
            centroids.append(emb)
            members.append([sid])
            counts.append(1)

    clusters: list[UnknownCluster] = []
    for idx, (mem, cnt) in enumerate(zip(members, counts)):
        if cnt < min_size:
            continue
        sample = _sample_ids(mem)
        clusters.append(UnknownCluster(
            cluster_id=idx,
            size=cnt,
            member_ids=mem,
            centroid=centroids[idx],
            best_named_match=None,
            sample_segment_ids=sample,
        ))
    clusters.sort(key=lambda c: -c.size)
    return clusters[:max_clusters]


def _sample_ids(ids: list[int], k: int = 5) -> list[int]:
    if len(ids) <= k:
        return list(ids)
    step = max(len(ids) // k, 1)
    return [ids[i] for i in range(0, len(ids), step)][:k]
