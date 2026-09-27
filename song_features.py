#!/usr/bin/env python3
"""
song_features.py
================

Extract the musical-feature table below from audio files using librosa.

    Index | Song Name | Artist | Year of Release | Streams | Song Length |
    Intro Length | Chorus Length | Repetition | Time to First Hook | Key |
    Chord Complexity | Melodic Range | Tempo | Syncopation | Loudness |
    Danceability | Dynamic Range | Distortion | Spectral Brightness

Accuracy notes (v2)
-------------------
This version tightens the parts that were making real mistakes:

* Key now uses tuning-corrected, log-compressed chroma, a vote across three
  published key profiles, and a separate bass chroma. Bass matters: E major and
  B major share six of seven notes, so the note the bass sits on is often the
  only reliable way to tell a key from its dominant.
* Tempo is estimated with a flat prior instead of librosa's default pull toward
  120 BPM, then refined from the actual tracked beat spacing.
* Chords are smoothed with a Viterbi pass, so flicker between adjacent frames
  stops inflating the chord-change rate.
* Loudness, dynamic range, brightness and distortion are measured at 44.1 kHz
  rather than the downsampled analysis signal, so nothing above 11 kHz is lost.
* Melodic range discards octave jumps before taking its percentiles.
* Structure detection runs a path-enhancement filter over the similarity
  matrix, which makes repeated sections stand out from the noise.

Run `python song_features.py --self-test` to check the extractor against
synthetic audio with known answers. It takes about a minute and tells you
whether key, tempo, brightness and loudness are being recovered correctly on
your machine.

Install
-------
    pip install librosa soundfile numpy scipy
    pip install mutagen pyloudnorm         # optional: tag reading + true LUFS
    pip install imageio-ffmpeg             # needed for .m4a/.aac/.wma

Usage
-----
    # a whole library, including every subfolder, .m4a files only
    python song_features.py ./library -o features.csv --ext m4a

    # the same again later: already-done songs are skipped, new ones appended
    python song_features.py ./library -o features.csv --ext m4a

    # faster for big batches
    python song_features.py ./library -o features.csv --workers 4 --pitch piptrack

    # start the spreadsheet over from scratch
    python song_features.py ./library -o features.csv --overwrite

By default the CSV is added to, not replaced, and songs already in it are
skipped -- so an interrupted run is resumed just by running it again.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import librosa  # noqa: E402

try:
    import scipy.ndimage as ndi  # type: ignore
except ImportError:
    ndi = None

try:
    import pyloudnorm  # type: ignore
except ImportError:
    pyloudnorm = None

try:
    import mutagen  # type: ignore
except ImportError:
    mutagen = None

try:  # ships a private ffmpeg binary, no system install or admin rights needed
    import imageio_ffmpeg  # type: ignore
except ImportError:
    imageio_ffmpeg = None


VERSION = "4.4"

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif"}

COLUMNS: List[str] = [
    "Index", "Song Name", "Artist", "Year of Release", "Streams", "Song Length",
    "Intro Length", "Chorus Length", "Repetition", "Time to First Hook", "Key",
    "Chord Complexity", "Melodic Range", "Tempo", "Syncopation", "Danceability",
    "Loudness", "Dynamic Range", "Distortion", "Spectral Brightness",
]

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Three published key-profile sets. Each has its own systematic biases, so the
# key finder scores all three and adds the results -- an error has to be shared
# by the majority to survive.
KEY_PROFILES: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    # Krumhansl & Kessler (1982), from listener probe-tone ratings
    "krumhansl": (
        np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
        np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]),
    ),
    # Temperley (1999), tuned on the Kostka-Payne corpus
    "temperley": (
        np.array([0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400]),
        np.array([0.712, 0.084, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330]),
    ),
    # Sha'ath (2011), derived for popular music in the KeyFinder project
    "shaath": (
        np.array([6.6, 2.0, 3.5, 2.3, 4.6, 4.0, 2.5, 5.2, 2.4, 3.7, 2.3, 3.4]),
        np.array([6.5, 2.7, 3.5, 5.4, 2.6, 3.5, 2.5, 5.2, 4.0, 2.7, 4.3, 3.2]),
    ),
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    sr: int = 22050              # analysis rate for pitch, beats and structure
    timbre_sr: int = 44100       # loudness/brightness/distortion are measured here
    hop_length: int = 512
    n_fft: int = 2048
    offset: float = 0.0
    duration: Optional[float] = None
    pitch: str = "pyin"          # "pyin" | "piptrack" | "none"
    min_intro: float = 3.0
    max_intro: float = 90.0
    min_chorus: float = 6.0
    max_chorus: float = 60.0
    bass_weight: float = 0.30    # how much the bass note steers key detection
    edge_weight: float = 0.30    # how much the opening/closing bars steer it
    chord_switch_penalty: float = 1.2   # higher = fewer, longer chords
    verbose: bool = True
    artist_from_folder: bool = False


# --------------------------------------------------------------------------- #
# Small numeric helpers (pure NumPy -- unit-testable without audio)
# --------------------------------------------------------------------------- #
def _safe(x: Any, ndigits: int = 4) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, ndigits) if math.isfinite(v) else None


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if w <= 1 or x.size == 0:
        return x
    w = int(min(w, x.size))
    return np.convolve(x, np.ones(w) / w, mode="same")


def contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Half-open [start, end) ranges of consecutive True values."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def normalized_entropy(counts: Sequence[float]) -> float:
    p = np.asarray(counts, dtype=float)
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)) / np.log(p.size))


def gaussian(x: float, mu: float, sigma: float) -> float:
    return float(np.exp(-0.5 * ((x - mu) / sigma) ** 2))


def fit_length(X: np.ndarray, n: int) -> np.ndarray:
    """Trim/edge-pad a (features, frames) array to exactly `n` frames."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    m = X.shape[1]
    if m == n:
        return X
    if m > n:
        return X[:, :n]
    pad = np.repeat(X[:, -1:] if m else np.zeros((X.shape[0], 1)), n - m, axis=1)
    return np.concatenate([X, pad], axis=1)


def log_compress(X: np.ndarray, gamma: float = 100.0) -> np.ndarray:
    """Logarithmic magnitude compression, as used for CENS-style chroma.

    Raw chroma is dominated by whichever notes happen to be loudest, which lets
    one belted chorus note outvote the harmony of the whole song. Compressing
    first makes quiet-but-persistent pitches count properly.
    """
    X = np.asarray(X, dtype=float)
    return np.log1p(gamma * np.maximum(X, 0.0))


def viterbi_path(scores: np.ndarray, switch_penalty: float) -> np.ndarray:
    """Best state sequence when changing state costs `switch_penalty`.

    `scores` is (states, frames), higher meaning a better fit. Without this,
    frame-by-frame chord picking flickers between near-tied candidates and
    massively over-counts chord changes.
    """
    S = np.asarray(scores, dtype=float)
    if S.ndim != 2 or S.shape[1] == 0:
        return np.zeros(0, dtype=int)
    n_states, n_frames = S.shape
    dp = np.empty_like(S)
    back = np.zeros(S.shape, dtype=int)
    dp[:, 0] = S[:, 0]
    idx = np.arange(n_states)
    for t in range(1, n_frames):
        prev = dp[:, t - 1]
        best_prev_val = float(prev.max())
        best_prev_idx = int(prev.argmax())
        switch_val = best_prev_val - switch_penalty
        stay_wins = prev >= switch_val
        dp[:, t] = np.where(stay_wins, prev, switch_val) + S[:, t]
        back[:, t] = np.where(stay_wins, idx, best_prev_idx)
    path = np.zeros(n_frames, dtype=int)
    path[-1] = int(dp[:, -1].argmax())
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = back[path[t], t]
    return path


def refine_tempo_from_beats(beat_times: np.ndarray, fallback: float,
                            tolerance: float = 0.15) -> float:
    """Recover a precise tempo by fitting a straight line to the beat times.

    The tempogram can only report tempos corresponding to whole-frame lags, so
    at the default hop a true 146 BPM lands on 143.5 simply because lags 17 and
    18 frames bracket it with nothing in between. The tracked beats don't have
    that problem in aggregate: they snap to real onsets, so regressing beat time
    against beat number recovers the spacing to well under a frame.

    Only the longest stretch of evenly-spaced beats is used, so a dropped or
    doubled beat can't tilt the line. Falls back to the tracker's own figure if
    the result disagrees by more than `tolerance`.
    """
    t = np.asarray(beat_times, dtype=float).ravel()
    if t.size < 8 or not math.isfinite(fallback) or fallback <= 0:
        return float(fallback)

    ibi = np.diff(t)
    med = float(np.median(ibi))
    if med <= 0:
        return float(fallback)

    steady = np.abs(ibi - med) <= 0.2 * med
    runs = contiguous_runs(steady)
    if not runs:
        return float(fallback)
    s, e = max(runs, key=lambda r: r[1] - r[0])
    if e - s < 6:                       # too few consecutive steady beats to fit
        return float(fallback)

    seg = t[s:e + 1]
    slope = float(np.polyfit(np.arange(seg.size, dtype=float), seg, 1)[0])
    if slope <= 0:
        return float(fallback)

    refined = 60.0 / slope
    if abs(refined - fallback) / fallback > tolerance:
        return float(fallback)          # disagreement means a half/double error
    return float(refined)


def remove_octave_jumps(midi: np.ndarray, window: int = 9, tol: float = 6.0) -> np.ndarray:
    """Drop pitch estimates that sit more than `tol` semitones from their
    local median -- almost always octave errors rather than real leaps."""
    m = np.asarray(midi, dtype=float)
    if m.size < window:
        return m
    if ndi is not None:
        med = ndi.median_filter(m, size=window, mode="nearest")
    else:
        pad = window // 2
        padded = np.pad(m, pad, mode="edge")
        med = np.array([np.median(padded[i:i + window]) for i in range(m.size)])
    return m[np.abs(m - med) <= tol]


# --------------------------------------------------------------------------- #
# Key detection
# --------------------------------------------------------------------------- #
def parse_chord(label: str) -> Optional[Tuple[int, str]]:
    """Split a chord name into (root pitch class, quality)."""
    if not label:
        return None
    m = re.match(r"^([A-G]#?)(.*)$", label)
    if not m or m.group(1) not in PITCH_CLASSES:
        return None
    return PITCH_CLASSES.index(m.group(1)), m.group(2)


def triad_template(tonic: int, mode: str) -> np.ndarray:
    """Unit-length chroma template for a tonic triad."""
    intervals = (0, 4, 7) if mode == "major" else (0, 3, 7)
    t = np.zeros(12)
    for i in intervals:
        t[(tonic + i) % 12] = 1.0
    return t / np.linalg.norm(t)


def triad_support(chroma_vec: Optional[np.ndarray], tonic: int, mode: str) -> float:
    """How strongly a chroma vector resembles a given tonic triad, 0-1.

    Threshold-free by design: unlike chord labelling, this always returns a
    usable number, so a tie-break built on it can never silently do nothing.
    """
    if chroma_vec is None:
        return 0.0
    v = np.asarray(chroma_vec, dtype=float).ravel()
    if v.size != 12:
        return 0.0
    n = float(np.linalg.norm(v))
    if n <= 0:
        return 0.0
    return float(np.dot(v / n, triad_template(tonic, mode)))


def relative_tiebreak_score(tonic: int, mode: str, labels: Sequence[Optional[str]]) -> float:
    """How much a chord sequence supports one key of a relative pair.

    Four pieces of evidence, in descending order of usefulness: the song opens
    on the tonic chord, it closes on the tonic chord, chords rooted on the tonic
    are common, and the tonic chord's own quality matches the mode. The first
    two matter most -- C-Am-F-G and Am-F-C-G contain identical notes and
    identical chords, and the only thing separating them is where they start.
    """
    parsed = [p for p in (parse_chord(c) for c in labels if c) if p]
    if not parsed:
        return 0.0

    roots = [r for r, _ in parsed]
    score = 0.0
    if roots[0] == tonic:
        score += 2.0
    if roots[-1] == tonic:
        score += 2.0
    score += 1.5 * (roots.count(tonic) / len(roots))

    qualities = [q for r, q in parsed if r == tonic]
    if qualities:
        minor_share = sum(1 for q in qualities if q.startswith("m")) / len(qualities)
        score += minor_share if mode == "minor" else (1.0 - minor_share)

    # A cadence term was tried here -- rewarding a tonic approached from its
    # dominant (V-I) or flat seventh (bVII-i). It was removed because it cannot
    # separate relative keys either: in G-D-Em-C, the D-Em move is an ordinary
    # IV-v within G major but reads as a textbook bVII-i into E minor. Chord
    # transitions are shared between relative keys just as their notes are.
    return score


def is_relative_pair(tonic_a: int, mode_a: str, tonic_b: int, mode_b: str) -> bool:
    """True for a relative major/minor pair, e.g. C major and A minor.

    The relative minor sits nine semitones above its major (or three below), and
    the two share an identical set of seven notes.
    """
    if mode_a == mode_b:
        return False
    major, minor = ((tonic_a, tonic_b) if mode_a == "major" else (tonic_b, tonic_a))
    return (minor - major) % 12 == 9


def _as_distribution(v: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if v is None:
        return None
    a = np.asarray(v, dtype=float).ravel()
    return a / a.sum() if (a.size == 12 and a.sum() > 0) else None


def estimate_key_from_chroma(
    chroma_mean: np.ndarray,
    bass_mean: Optional[np.ndarray] = None,
    head_mean: Optional[np.ndarray] = None,
    tail_mean: Optional[np.ndarray] = None,
    chord_labels: Optional[Sequence[Optional[str]]] = None,
    bass_weight: float = 0.30,
    edge_weight: float = 0.30,
) -> Dict[str, Any]:
    """Estimate musical key by voting across three published profile sets,
    then breaking ties with the bass line and the song's opening/closing bars.

    Template correlation alone has two signature failures. It confuses a key
    with its dominant -- E and B major differ by a single note -- and it
    confuses a key with its relative minor, since C major and A minor use
    exactly the same seven notes.

    The bass fixes the first: the bass sits on the tonic far more than on the
    dominant. The edges fix the second: C-Am-F-G and Am-F-C-G contain identical
    pitch content, and the only thing distinguishing them is which chord the
    music starts and ends on. `edge_mean` should be chroma averaged over the
    first and last few percent of the track, with the ending weighted higher.

    Returns the winning key plus the runner-up and a confidence gap, so you can
    filter or hand-check the uncertain ones.
    """
    v = np.asarray(chroma_mean, dtype=float).ravel()
    unknown = {"key": "unknown", "tonic": "unknown", "mode": "unknown",
               "confidence": 0.0, "runner_up": "unknown"}
    if v.size != 12 or not np.any(v > 0):
        return unknown

    vc = v - v.mean()
    v_norm = float(np.linalg.norm(vc))
    if v_norm <= 0:
        return unknown

    b = _as_distribution(bass_mean)
    edges = [x for x in (head_mean, tail_mean) if x is not None]
    e = _as_distribution(np.mean(edges, axis=0)) if edges else None

    totals: Dict[Tuple[int, str], float] = {}
    for major, minor in KEY_PROFILES.values():
        for mode, profile in (("major", major), ("minor", minor)):
            p = profile - profile.mean()
            p_norm = float(np.linalg.norm(p))
            for tonic in range(12):
                r = float(np.dot(vc, np.roll(p, tonic))) / (v_norm * p_norm)
                totals[(tonic, mode)] = totals.get((tonic, mode), 0.0) + r

    n_profiles = len(KEY_PROFILES)
    flat = 1.0 / 12.0
    scored = []
    for (tonic, mode), total in totals.items():
        score = total / n_profiles                      # mean correlation, -1..1
        if b is not None:
            score += bass_weight * (float(b[tonic]) - flat) * 3.0
        if e is not None:
            score += edge_weight * (float(e[tonic]) - flat) * 3.0
        scored.append((score, tonic, mode))
    scored.sort(key=lambda s: s[0], reverse=True)
    best, second = scored[0], scored[1]
    confidence = _clip01((best[0] - second[0]) / 0.25)

    # Relative major/minor tie-break. C major and A minor contain exactly the
    # same seven notes, so whole-song pitch content cannot separate them even in
    # principle -- whichever wins, wins by profile quirk. When the top two are
    # such a pair, throw away the profile scores and decide purely on where the
    # bass sits and how the track opens and closes, which is the only evidence
    # that actually distinguishes them.
    if is_relative_pair(best[1], best[2], second[1], second[2]):
        # Prefer the chord sequence when we actually have one. Chord labels are
        # categorical -- "the song opens on Am" is a fact, not a correlation --
        # and they already fold in the opening chord, the closing chord, how
        # often the tonic is played, and whether it is played major or minor.
        # Raw chroma correlation is the fallback for when labelling comes up
        # empty, and it must not outvote the labels when both are present.
        labelled = sum(1 for c in (chord_labels or []) if c)
        use_chords = labelled >= max(4, 0.25 * len(chord_labels or []))

        def support(tonic: int, mode: str) -> float:
            if use_chords:
                return relative_tiebreak_score(tonic, mode, chord_labels)
            # Take the STRONGER of the opening and closing bar, not their sum.
            # Summing lets the two cancel: a loop that starts on Am and ends on
            # G gives C major exactly as much support as A minor, because G is
            # the V of C. The real question is whether the track opens *or*
            # closes on the tonic triad.
            s = (2.0 * max(triad_support(head_mean, tonic, mode),
                           triad_support(tail_mean, tonic, mode))
                 + 1.0 * triad_support(v, tonic, mode))
            if b is not None:
                s += bass_weight * 3.0 * float(b[tonic])
            return s

        s_best = support(best[1], best[2])
        s_second = support(second[1], second[2])
        if s_second > s_best:
            best, second = second, best
            s_best, s_second = s_second, s_best
        confidence = min(confidence, _clip01(abs(s_best - s_second) / 0.5))
        tiebreak = {"relative_tiebreak": True,
                    "tiebreak_margin": round(float(s_best - s_second), 4)}
    else:
        tiebreak = {"relative_tiebreak": False, "tiebreak_margin": None}

    result = {
        "key": f"{PITCH_CLASSES[best[1]]} {best[2]}",
        "tonic": PITCH_CLASSES[best[1]],
        "mode": best[2],
        "confidence": confidence,
        "runner_up": f"{PITCH_CLASSES[second[1]]} {second[2]}",
    }
    result.update(tiebreak)
    return result


# --------------------------------------------------------------------------- #
# Chords
# --------------------------------------------------------------------------- #
def build_chord_templates() -> Tuple[np.ndarray, List[str]]:
    """Major, minor, and dominant-7th templates for all 12 roots.

    Sevenths are included because pop and rock lean on them constantly; without
    them a G7 gets mislabelled as a B-diminished-ish near-match and inflates the
    apparent chord vocabulary.
    """
    shapes = [("", (0, 4, 7)), ("m", (0, 3, 7)), ("7", (0, 4, 7, 10))]
    templates, names = [], []
    for root in range(12):
        for suffix, intervals in shapes:
            t = np.zeros(12)
            for i in intervals:
                t[(root + i) % 12] = 1.0
            templates.append(t / np.linalg.norm(t))
            names.append(f"{PITCH_CLASSES[root]}{suffix}")
    return np.array(templates), names


def chord_sequence_from_chroma(
    beat_chroma: np.ndarray,
    min_strength: float = 0.45,
    switch_penalty: float = 1.2,
    flat_floor: float = 0.08,
) -> Tuple[List[Optional[str]], np.ndarray]:
    """Label each beat with a chord, smoothed so it doesn't flicker.

    Frames whose best match is weaker than `min_strength` are left unlabelled --
    silence, noise, or genuinely non-triadic harmony.
    """
    templates, names = build_chord_templates()
    X = np.asarray(beat_chroma, dtype=float)
    if X.ndim != 2 or X.shape[0] != 12 or X.shape[1] == 0:
        return [], np.zeros(0)

    # Match on SHAPE, not level. Real chroma always has a leakage floor in every
    # bin, and log compression lifts it further, so a raw cosine against a
    # sparse triad template scores worse than a flat "any note" template does --
    # which silently leaves every beat unlabelled. Centring both sides removes
    # that common floor and compares only the peaks-above-average pattern.
    Xc = X - X.mean(axis=0, keepdims=True)
    norms_c = np.linalg.norm(Xc, axis=0)
    norms_raw = np.linalg.norm(X, axis=0)

    Tc = templates - templates.mean(axis=1, keepdims=True)
    Tc = Tc / np.linalg.norm(Tc, axis=1, keepdims=True)

    sims = Tc @ (Xc / np.maximum(norms_c, 1e-9))         # (n_templates, n_beats)

    # A genuinely chord-less beat is flat, so almost all of its energy sits in
    # the mean and very little in the shape. That ratio is the "no chord" test.
    shape = norms_c / np.maximum(norms_raw, 1e-9)

    path = viterbi_path(sims, switch_penalty=switch_penalty)
    chosen = sims[path, np.arange(sims.shape[1])]
    labels = [names[i] if (s >= min_strength and sh >= flat_floor) else None
              for i, s, sh in zip(path, chosen, shape)]
    return labels, chosen


def chord_complexity_score(
    labels: Sequence[Optional[str]], duration_sec: float
) -> Tuple[float, Dict[str, float]]:
    """0-1 harmonic-complexity score: vocabulary, evenness and change rate."""
    clean = [c for c in labels if c]
    if not clean or duration_sec <= 0:
        return 0.0, {"unique_chords": 0.0, "chord_changes_per_min": 0.0, "chord_entropy": 0.0}

    unique = sorted(set(clean))
    counts = [clean.count(c) for c in unique]
    changes = sum(1 for a, b in zip(clean[:-1], clean[1:]) if a != b)
    changes_per_min = changes / (duration_sec / 60.0)

    vocabulary = _clip01(len(unique) / 12.0)
    evenness = normalized_entropy(counts)
    rate = _clip01(changes_per_min / 30.0)
    return _clip01((vocabulary + evenness + rate) / 3.0), {
        "unique_chords": float(len(unique)),
        "chord_changes_per_min": float(changes_per_min),
        "chord_entropy": float(evenness),
    }


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #
def find_repeated_segment(
    rec: np.ndarray,
    min_lag: int = 8,
    min_len: int = 8,
    max_len: Optional[int] = None,
    smooth: int = 4,
    weights: Optional[np.ndarray] = None,
) -> Optional[Tuple[int, int, int, float]]:
    """Locate the most prominent repeated block in a self-similarity matrix.

    A section that repeats `lag` frames later shows up as a bright diagonal
    stripe at that offset. Returns (start, length, lag, score) in frame units.
    """
    R = np.asarray(rec, dtype=float)
    n = R.shape[0]
    if n < min_lag + min_len:
        return None

    best: Optional[Tuple[int, int, int, float]] = None
    for lag in range(min_lag, n - min_len):
        diag = np.diagonal(R, offset=-lag)          # diag[t] == R[t + lag, t]
        if diag.size < min_len:
            continue
        d = moving_average(diag, smooth)
        thr = max(0.15, float(np.percentile(d, 70)))
        for s, e in contiguous_runs(d >= thr):
            length = e - s
            if length < min_len:
                continue
            if max_len is not None:
                length = min(length, max_len)
                e = s + length
            w = 1.0 if weights is None else float(np.mean(weights[s:e]) + 1e-6)
            score = length * float(np.mean(d[s:e])) * w
            if best is None or score > best[3]:
                best = (int(s), int(length), int(lag), float(score))
    return best


def syncopation_from_grid(grid_strength: np.ndarray, subdivisions: int = 4) -> float:
    """0-1 share of onset energy landing on metrically weak grid positions."""
    e = np.maximum(np.asarray(grid_strength, dtype=float), 0.0)
    if e.size == 0 or e.sum() <= 0:
        return 0.0
    if subdivisions == 4:
        weights = np.array([1.0, 0.25, 0.5, 0.25])
    elif subdivisions == 2:
        weights = np.array([1.0, 0.4])
    else:
        weights = np.linspace(1.0, 0.25, subdivisions)
    w = np.tile(weights, int(np.ceil(e.size / subdivisions)))[: e.size]
    return _clip01(float(np.sum(e * (1.0 - w)) / np.sum(e)))


# --------------------------------------------------------------------------- #
# Audio loading
# --------------------------------------------------------------------------- #
def true_duration(path: os.PathLike | str) -> Optional[float]:
    """The track's real length in seconds, without decoding the whole file.

    Used by the web app: it caps how much audio the expensive per-frame
    analysis (melody tracking, chroma, structure detection) actually looks
    at, for speed -- but the model still needs the track's REAL length, not
    the length of the analysis window. `librosa.get_duration` reads this from
    the file directly (header / quick scan), not by running the full DSP
    pipeline, so it stays fast even when the capped analysis above it is
    only looking at the first couple of minutes.
    """
    try:
        return float(librosa.get_duration(path=str(path)))
    except Exception:
        return None


def find_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None
    return None


def load_audio(
    path: Path, sr: int, offset: float = 0.0, duration: Optional[float] = None
) -> Tuple[np.ndarray, int]:
    """Decode any audio file to a mono float32 array at `sr`.

    librosa handles WAV/FLAC/OGG and most MP3s. Formats it can't open --
    .m4a/AAC, .wma -- go through ffmpeg, read straight from its stdout.
    """
    try:
        y, sr_out = librosa.load(str(path), sr=sr, mono=True, offset=offset, duration=duration)
        if y.size:
            return y, int(sr_out)
    except Exception:
        pass

    exe = find_ffmpeg()
    if exe is None:
        raise RuntimeError(
            f"Could not decode {path.name}. This format needs ffmpeg -- run "
            f"'pip install imageio-ffmpeg' and try again."
        )

    cmd = [exe, "-nostdin", "-v", "error"]
    if offset:
        cmd += ["-ss", str(offset)]
    cmd += ["-i", str(path)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-"]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        msg = proc.stderr.decode("utf-8", "replace").strip()[:400]
        raise RuntimeError(f"ffmpeg failed to decode {path.name}: {msg}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy(), sr


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def read_tags(path: Path, folder_artist: bool = False) -> Dict[str, Optional[str]]:
    """Song name / artist / year from embedded tags, falling back to filename."""
    info: Dict[str, Optional[str]] = {"title": None, "artist": None, "year": None}

    if mutagen is not None:
        try:
            f = mutagen.File(str(path), easy=True)
            if f is not None and f.tags:
                def first(*keys: str) -> Optional[str]:
                    for k in keys:
                        v = f.tags.get(k)
                        if v:
                            return str(v[0]) if isinstance(v, list) else str(v)
                    return None
                info["title"] = first("title")
                info["artist"] = first("artist", "albumartist")
                raw_year = first("date", "originaldate", "year")
                if raw_year:
                    m = re.search(r"(19|20)\d{2}", raw_year)
                    info["year"] = m.group(0) if m else raw_year
        except Exception:
            pass

    if not info["title"]:
        stem = path.stem
        parts = [p.strip() for p in re.split(r"\s+-\s+", stem) if p.strip()]
        parts = [p for p in parts if not re.fullmatch(r"\d{1,3}", p)]
        if len(parts) >= 2:
            info["artist"] = info["artist"] or parts[0]
            info["title"] = parts[-1]
        else:
            info["title"] = stem

    if folder_artist and not info["artist"] and path.parent.name:
        info["artist"] = path.parent.name
    return info


# --------------------------------------------------------------------------- #
# The extractor
# --------------------------------------------------------------------------- #
class SongFeatureExtractor:
    """Computes every audio-derived column for one file.

    Two sample rates are used on purpose. Pitch, beats and structure run at
    22.05 kHz, which is plenty and keeps the expensive steps affordable.
    Loudness, dynamic range, brightness and distortion run at 44.1 kHz, because
    downsampling throws away everything above 11 kHz -- exactly the cymbals and
    air those four features are supposed to be measuring.
    """

    def __init__(self, path: os.PathLike | str, cfg: Optional[Config] = None,
                 audio: Optional[Tuple[np.ndarray, int]] = None):
        self.path = Path(path)
        self.cfg = cfg or Config()
        self._injected = audio          # used by --self-test
        self.diag: Dict[str, Any] = {}

    def _log(self, msg: str) -> None:
        if self.cfg.verbose:
            print(f"      ... {msg}", file=sys.stderr, flush=True)

    # -- shared analysis ---------------------------------------------------- #
    def _prepare(self) -> None:
        cfg = self.cfg
        self._log("decoding audio")
        if self._injected is not None:
            self.y_hi, self.sr_hi = self._injected
        else:
            self.y_hi, self.sr_hi = load_audio(
                self.path, cfg.timbre_sr, cfg.offset, cfg.duration
            )
        if self.y_hi.size == 0:
            raise ValueError("decoded 0 samples -- unreadable or empty audio")

        self.duration = float(len(self.y_hi) / self.sr_hi)
        self.hop = cfg.hop_length

        # Full-rate spectrogram for the timbre features.
        hop_hi = self.hop * 2
        self.S_hi = np.abs(librosa.stft(self.y_hi, n_fft=cfg.n_fft * 2, hop_length=hop_hi))
        self.freqs_hi = librosa.fft_frequencies(sr=self.sr_hi, n_fft=cfg.n_fft * 2)

        # Downsampled signal for everything else.
        if self.sr_hi != cfg.sr:
            self.y = librosa.resample(self.y_hi, orig_sr=self.sr_hi, target_sr=cfg.sr)
        else:
            self.y = self.y_hi
        self.sr = cfg.sr

        self._log(f"{self.duration:.1f}s decoded; separating harmonic and percussive parts")
        self.y_harm, self.y_perc = librosa.effects.hpss(self.y)

        self.rms = librosa.feature.rms(
            y=self.y_hi, frame_length=cfg.n_fft * 2, hop_length=hop_hi
        )[0]
        self.rms_db = librosa.amplitude_to_db(np.maximum(self.rms, 1e-10), ref=1.0)

        self.onset_env = librosa.onset.onset_strength(
            y=self.y_perc, sr=self.sr, hop_length=self.hop, aggregate=np.median
        )

        self._log("tracking beats and tempo")
        self.tempo, self.beats = self._track_tempo()
        self.beat_times = librosa.frames_to_time(self.beats, sr=self.sr, hop_length=self.hop)

        # Tuning matters: a recording a quarter-tone flat smears every chroma
        # bin, which is one of the quiet killers of key and chord accuracy.
        try:
            self.tuning = float(librosa.estimate_tuning(y=self.y_harm, sr=self.sr))
        except Exception:
            self.tuning = 0.0
        self.diag["tuning_offset_semitones"] = self.tuning

        self._log(f"tempo {self.tempo:.1f} BPM; computing chroma")
        self.chroma = self._chroma(self.y_harm)
        self.bass_chroma = self._chroma(
            self.y_harm, fmin=librosa.note_to_hz("C1"), n_octaves=3
        )

        n_frames = self.chroma.shape[1]
        bounds = librosa.util.fix_frames(self.beats, x_min=0, x_max=n_frames)
        self.beat_chroma = librosa.util.sync(self.chroma, bounds, aggregate=np.median)
        n_seg = self.beat_chroma.shape[1]
        self.seg_times = librosa.frames_to_time(
            bounds, sr=self.sr, hop_length=self.hop
        )[: n_seg + 1]

        rms_lo = librosa.feature.rms(y=self.y, frame_length=cfg.n_fft, hop_length=self.hop)[0]
        beat_rms = librosa.util.sync(
            fit_length(rms_lo[np.newaxis, :], n_frames), bounds, aggregate=np.mean
        )[0]
        self.beat_energy = beat_rms[:n_seg]
        if self.beat_energy.size and self.beat_energy.max() > 0:
            self.beat_energy = self.beat_energy / self.beat_energy.max()

        self.mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13, hop_length=self.hop)
        self.beat_mfcc = librosa.util.sync(
            fit_length(self.mfcc, n_frames), bounds, aggregate=np.mean
        )[:, :n_seg]

    def _chroma(self, y: np.ndarray, fmin: Optional[float] = None,
                n_octaves: int = 7) -> np.ndarray:
        """Tuning-corrected, log-compressed CQT chroma."""
        kwargs: Dict[str, Any] = dict(
            y=y, sr=self.sr, hop_length=self.hop, tuning=self.tuning, n_octaves=n_octaves
        )
        if fmin is not None:
            kwargs["fmin"] = fmin
        try:
            C = librosa.feature.chroma_cqt(**kwargs)
        except Exception:
            kwargs.pop("n_octaves", None)
            kwargs.pop("fmin", None)
            C = librosa.feature.chroma_cqt(**kwargs)
        C = log_compress(C)
        # Normalise per frame so loud moments don't outvote the rest of the song.
        norms = np.linalg.norm(C, axis=0, keepdims=True)
        return C / np.maximum(norms, 1e-9)

    def _track_tempo(self) -> Tuple[float, np.ndarray]:
        """Tempo with a flat prior, refined from the tracked beat spacing.

        librosa's default beat tracker leans on a prior centred at 120 BPM,
        which drags fast songs slow and slow songs fast. We first estimate the
        tempo with that prior switched off, then track beats seeded with that
        value, then take the median beat-to-beat spacing as the final answer.
        """
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=self.onset_env, sr=self.sr, hop_length=self.hop, units="frames"
        )
        tempo = float(np.atleast_1d(tempo)[0])
        beats = np.asarray(beats, dtype=int)

        times = librosa.frames_to_time(beats, sr=self.sr, hop_length=self.hop)
        refined = refine_tempo_from_beats(times, tempo)
        self.diag["tempo_from_tempogram"] = tempo
        self.diag["tempo_from_beats"] = refined
        if times.size > 2:
            ibi = np.diff(times)
            self.diag["beat_ibi_cv"] = float(np.std(ibi) / max(np.mean(ibi), 1e-9))
        return refined, beats

    # -- structure ---------------------------------------------------------- #
    def intro_length(self) -> float:
        """Seconds until the first structural boundary."""
        cfg = self.cfg
        candidates: List[float] = []

        n_seg = self.beat_chroma.shape[1]
        if n_seg >= 16:
            feats = np.vstack([
                librosa.util.normalize(self.beat_chroma, axis=0),
                librosa.util.normalize(self.beat_mfcc, axis=0),
            ])
            k = int(np.clip(round(self.duration / 20.0), 4, 12))
            try:
                bounds = librosa.segment.agglomerative(feats, k)
                btimes = self.seg_times[np.clip(bounds, 0, len(self.seg_times) - 1)]
                for t in np.sort(btimes):
                    if cfg.min_intro <= t <= min(cfg.max_intro, 0.5 * self.duration):
                        candidates.append(float(t))
                        break
            except Exception:
                pass

        if self.rms.size:
            thresh = 0.5 * float(np.median(self.rms))
            hop_hi = self.hop * 2
            hold = max(1, int(self.sr_hi / hop_hi))
            for s, e in contiguous_runs(self.rms > thresh):
                if e - s >= hold:
                    t = float(librosa.frames_to_time(s, sr=self.sr_hi, hop_length=hop_hi))
                    if cfg.min_intro <= t <= cfg.max_intro:
                        candidates.append(t)
                    break

        self.diag["intro_candidates"] = candidates
        return float(min(candidates)) if candidates else 0.0

    def _recurrence(self) -> Optional[np.ndarray]:
        """Beat-synchronous self-similarity, with diagonal paths enhanced."""
        n = self.beat_chroma.shape[1]
        if n < 24:
            return None
        stacked = librosa.feature.stack_memory(
            librosa.util.normalize(self.beat_chroma, axis=0), n_steps=4, delay=2
        )
        try:
            R = librosa.segment.recurrence_matrix(
                stacked, width=3, mode="affinity", metric="cosine", sym=True
            )
        except Exception:
            return None
        # Path enhancement sharpens the diagonal stripes that mark real repeats
        # and suppresses the speckle that makes short false matches look real.
        try:
            R = librosa.segment.path_enhance(R, n=15, window="hann")
        except Exception:
            pass
        return R

    def structure(self) -> Dict[str, float]:
        out = {"chorus_length": 0.0, "time_to_first_hook": 0.0, "repetition": 0.0}
        R = self._recurrence()
        if R is None:
            return out

        density = float(np.mean(R > 0.1))
        out["repetition"] = _clip01(density / 0.25)
        self.diag["recurrence_density"] = density

        beats_per_sec = len(self.beat_times) / max(self.duration, 1e-9)
        min_len = max(4, int(self.cfg.min_chorus * beats_per_sec))
        max_len = max(min_len + 1, int(self.cfg.max_chorus * beats_per_sec))
        min_lag = max(8, int(8.0 * beats_per_sec))

        found = find_repeated_segment(
            R, min_lag=min_lag, min_len=min_len, max_len=max_len,
            smooth=4, weights=self.beat_energy,
        )
        if found is None:
            return out

        start, length, lag, score = found
        self.diag["chorus_lag_beats"] = lag
        self.diag["chorus_score"] = score
        last = len(self.seg_times) - 1
        t0 = float(self.seg_times[min(start, last)])
        t1 = float(self.seg_times[min(start + length, last)])
        out["chorus_length"] = max(0.0, t1 - t0)
        out["time_to_first_hook"] = t0
        return out

    # -- harmony / melody --------------------------------------------------- #
    def _head_tail_chroma(self, n_beats: int = 4) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Mean chroma over the opening and closing bar.

        Popular music tends to begin and end on the tonic, and that is the only
        information that can separate a key from its relative minor -- the two
        contain an identical set of notes across the song as a whole. One bar is
        used rather than a percentage of the track, so the window lands on a
        single chord instead of smearing across several.
        """
        B = self.beat_chroma
        if B.shape[1] >= 2 * n_beats:
            return B[:, :n_beats].mean(axis=1), B[:, -n_beats:].mean(axis=1)
        n = self.chroma.shape[1]
        if n < 8:
            return None, None
        k = max(1, n // 10)
        return self.chroma[:, :k].mean(axis=1), self.chroma[:, -k:].mean(axis=1)

    def chords(self) -> Tuple[List[Optional[str]], np.ndarray]:
        """Beat-by-beat chord labels, computed once and shared by key and
        chord-complexity."""
        if not hasattr(self, "_chords"):
            self._chords = chord_sequence_from_chroma(
                self.beat_chroma, switch_penalty=self.cfg.chord_switch_penalty
            )
        return self._chords

    def key(self) -> str:
        labels, _ = self.chords()
        head, tail = self._head_tail_chroma()
        result = estimate_key_from_chroma(
            np.mean(self.chroma, axis=1),
            bass_mean=np.mean(self.bass_chroma, axis=1),
            head_mean=head, tail_mean=tail,
            chord_labels=labels,
            bass_weight=self.cfg.bass_weight,
            edge_weight=self.cfg.edge_weight,
        )
        self.diag["key_confidence"] = result["confidence"]
        self.diag["key_runner_up"] = result["runner_up"]
        self.diag["key_mode"] = result["mode"]
        self.diag["relative_tiebreak"] = result["relative_tiebreak"]
        self.diag["tiebreak_margin"] = result["tiebreak_margin"]
        self.diag["labelled_beats"] = int(sum(1 for c in labels if c))
        self.diag["total_beats"] = int(len(labels))
        return result["key"]

    def chord_complexity(self) -> float:
        labels, sims = self.chords()
        score, extra = chord_complexity_score(labels, self.duration)
        self.diag["first_chord"] = next((c for c in labels if c), None)
        self.diag["last_chord"] = next((c for c in reversed(labels) if c), None)
        self.diag.update(extra)
        self.diag["chord_match_strength"] = float(np.mean(sims)) if sims.size else 0.0
        return score

    def melodic_range(self) -> float:
        """Pitch range of the leading melody, in semitones."""
        mode = self.cfg.pitch
        if mode == "none":
            return float("nan")
        self._log(f"tracking melody pitch ({mode}) -- this is the slow step")

        fmin = float(librosa.note_to_hz("C2"))
        fmax = float(librosa.note_to_hz("C7"))

        if mode == "pyin":
            f0, voiced, voiced_prob = librosa.pyin(
                self.y_harm, fmin=fmin, fmax=fmax, sr=self.sr,
                frame_length=2048, hop_length=self.hop,
            )
            good = np.isfinite(f0)
            if voiced_prob is not None:
                good &= np.nan_to_num(voiced_prob) > 0.5   # keep only confident frames
            f0 = f0[good]
        else:
            pitches, mags = librosa.piptrack(
                y=self.y_harm, sr=self.sr, hop_length=self.hop, fmin=fmin, fmax=fmax
            )
            idx = mags.argmax(axis=0)
            cols = np.arange(pitches.shape[1])
            f0, strength = pitches[idx, cols], mags[idx, cols]
            f0 = f0[(f0 > 0) & (strength > np.percentile(strength, 60))]

        if f0.size < 10:
            return float("nan")

        midi = remove_octave_jumps(librosa.hz_to_midi(f0))
        if midi.size < 10:
            return float("nan")
        lo, hi = np.percentile(midi, [5, 95])
        self.diag["median_pitch_midi"] = float(np.median(midi))
        self.diag["voiced_frames"] = int(midi.size)
        return float(hi - lo)

    # -- rhythm ------------------------------------------------------------- #
    def syncopation(self, subdivisions: int = 4) -> float:
        if self.beat_times.size < 4:
            return 0.0
        grid: List[float] = []
        for b0, b1 in zip(self.beat_times[:-1], self.beat_times[1:]):
            for k in range(subdivisions):
                grid.append(b0 + (b1 - b0) * k / subdivisions)
        frames = np.clip(
            librosa.time_to_frames(np.array(grid), sr=self.sr, hop_length=self.hop),
            0, len(self.onset_env) - 1,
        )
        return syncopation_from_grid(self.onset_env[frames], subdivisions)

    def danceability(self) -> float:
        """0-1 proxy: a strong, steady, mid-tempo, bass-driven pulse."""
        env = self.onset_env - self.onset_env.mean()
        pulse = 0.0
        if env.size > 4 and np.any(env):
            ac = librosa.autocorrelate(env, max_size=min(env.size, 4 * self.sr // self.hop))
            if ac[0] > 0:
                ac = ac / ac[0]
                if self.tempo > 0:
                    period = int(round((60.0 / self.tempo) * self.sr / self.hop))
                    if 1 < period < ac.size:
                        lo = max(1, int(period * 0.9))
                        hi = min(ac.size, int(period * 1.1) + 1)
                        pulse = float(np.max(ac[lo:hi]))
        pulse = _clip01(pulse)

        regularity = 0.0
        if self.beat_times.size > 4:
            ibi = np.diff(self.beat_times)
            if ibi.mean() > 0:
                regularity = _clip01(1.0 - float(np.std(ibi) / np.mean(ibi)) * 4.0)

        t = self.tempo if self.tempo > 0 else 120.0
        while t < 70:
            t *= 2
        while t > 180:
            t /= 2
        tempo_fit = gaussian(t, 120.0, 45.0)

        low = self.freqs_hi < 200.0
        low_ratio = float(self.S_hi[low].sum() / (float(self.S_hi.sum()) + 1e-9))
        drive = _clip01(low_ratio / 0.35)

        self.diag.update({"pulse_clarity": pulse, "beat_regularity": regularity,
                          "tempo_fit": tempo_fit, "low_end_ratio": low_ratio})
        return _clip01(0.35 * pulse + 0.30 * regularity + 0.20 * tempo_fit + 0.15 * drive)

    # -- loudness / timbre (all measured at the full sample rate) ----------- #
    def loudness(self) -> float:
        """Integrated LUFS if pyloudnorm is installed, else mean RMS dBFS."""
        if pyloudnorm is not None:
            try:
                lufs = float(pyloudnorm.Meter(self.sr_hi).integrated_loudness(self.y_hi))
                if math.isfinite(lufs):
                    self.diag["loudness_unit"] = "LUFS"
                    return lufs
            except Exception:
                pass
        self.diag["loudness_unit"] = "dBFS(RMS)"
        return float(np.mean(self.rms_db))

    def dynamic_range(self) -> float:
        """95th minus 10th percentile of frame RMS, in dB."""
        if self.rms_db.size == 0:
            return float("nan")
        p95, p10 = np.percentile(self.rms_db, [95, 10])
        peak_db = float(librosa.amplitude_to_db(np.array([np.max(np.abs(self.y_hi))]))[0])
        self.diag["crest_factor_db"] = peak_db - float(np.mean(self.rms_db))
        return float(p95 - p10)

    def distortion(self) -> float:
        """0-1 proxy for saturation: clipping, spectral flatness, low crest."""
        clip_ratio = float(np.mean(np.abs(self.y_hi) > 0.99))
        clip_term = _clip01(clip_ratio / 0.01)

        flatness = float(np.mean(librosa.feature.spectral_flatness(S=self.S_hi)[0]))
        flat_term = _clip01(flatness / 0.10)

        peak = float(np.max(np.abs(self.y_hi))) + 1e-9
        rms_all = float(np.sqrt(np.mean(self.y_hi ** 2))) + 1e-9
        crest_db = 20.0 * math.log10(peak / rms_all)
        crest_term = _clip01((14.0 - crest_db) / 8.0)

        self.diag.update({"clip_ratio": clip_ratio, "spectral_flatness": flatness,
                          "crest_db": crest_db})
        return _clip01(0.45 * clip_term + 0.30 * flat_term + 0.25 * crest_term)

    def spectral_brightness(self) -> float:
        """Spectral centroid in Hz, measured at 44.1 kHz."""
        centroid = librosa.feature.spectral_centroid(S=self.S_hi, sr=self.sr_hi)[0]
        rolloff = librosa.feature.spectral_rolloff(S=self.S_hi, sr=self.sr_hi, roll_percent=0.85)[0]
        high = self.freqs_hi >= 3000.0
        self.diag["spectral_rolloff85_hz"] = float(np.mean(rolloff))
        self.diag["high_freq_ratio"] = float(self.S_hi[high].sum() / (float(self.S_hi.sum()) + 1e-9))
        return float(np.mean(centroid))

    # -- driver ------------------------------------------------------------- #
    def extract(self, index: Optional[int] = None) -> Dict[str, Any]:
        self._prepare()
        tags = read_tags(self.path, folder_artist=self.cfg.artist_from_folder)
        self._log("analysing song structure (intro, chorus, repetition)")
        struct = self.structure()
        row: Dict[str, Any] = {
            "Index": index,
            "Song Name": tags["title"],
            "Artist": tags["artist"],
            "Year of Release": tags["year"],
            "Streams": None,
            "Song Length": _safe(self.duration, 2),
            "Intro Length": _safe(self.intro_length(), 2),
            "Chorus Length": _safe(struct["chorus_length"], 2),
            "Repetition": _safe(struct["repetition"], 3),
            "Time to First Hook": _safe(struct["time_to_first_hook"], 2),
            "Key": self.key(),
            "Chord Complexity": _safe(self.chord_complexity(), 3),
            "Melodic Range": _safe(self.melodic_range(), 2),
            "Tempo": _safe(self.tempo, 2),
            "Syncopation": _safe(self.syncopation(), 3),
            "Danceability": _safe(self.danceability(), 3),
            "Loudness": _safe(self.loudness(), 2),
            "Dynamic Range": _safe(self.dynamic_range(), 2),
            "Distortion": _safe(self.distortion(), 3),
            "Spectral Brightness": _safe(self.spectral_brightness(), 1),
        }
        self._log("done")
        row["_file"] = str(self.path)
        row["_diagnostics"] = {
            k: (_safe(v, 4) if isinstance(v, (int, float)) else v)
            for k, v in self.diag.items()
        }
        return row


def extract_features(path: os.PathLike | str, cfg: Optional[Config] = None,
                     index: Optional[int] = None) -> Dict[str, Any]:
    return SongFeatureExtractor(path, cfg).extract(index=index)


# --------------------------------------------------------------------------- #
# Self-test: synthesise audio with known answers and check we recover them
# --------------------------------------------------------------------------- #
def _synth_progression(chords: Sequence[Sequence[str]], bpm: float = 120.0,
                       beats_per_chord: int = 4, repeats: int = 4,
                       sr: int = 44100, bass_octave: str = "2") -> np.ndarray:
    """Render a chord progression as plucked tones with a root bass note."""
    beat = 60.0 / bpm
    seg_len = int(sr * beat * beats_per_chord)
    t = np.arange(seg_len) / sr
    out = []
    for _ in range(repeats):
        for chord in chords:
            seg = np.zeros(seg_len)
            root = chord[0]
            for note in chord:
                f = float(librosa.note_to_hz(note))
                for h, amp in ((1, 1.0), (2, 0.35), (3, 0.15)):
                    seg += amp * np.sin(2 * np.pi * f * h * t)
            fb = float(librosa.note_to_hz(root[:-1] + bass_octave))
            seg += 1.6 * np.sin(2 * np.pi * fb * t)
            # per-beat attack/decay so the beat tracker has something to find
            env = np.zeros(seg_len)
            per = seg_len // beats_per_chord
            for b in range(beats_per_chord):
                n = min(per, seg_len - b * per)
                env[b * per:b * per + n] = np.exp(-4.0 * np.arange(n) / sr / beat)
            out.append(seg * env)
    y = np.concatenate(out)
    return 0.4 * y / (np.max(np.abs(y)) + 1e-9)


def run_self_test() -> int:
    """Check the extractor against synthetic audio with known properties."""
    sr = 44100
    cfg = Config(verbose=False, pitch="piptrack")
    results: List[Tuple[bool, str, str]] = []

    def check(ok: bool, name: str, detail: str) -> None:
        results.append((bool(ok), name, detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name:<34} {detail}", flush=True)

    def run(y: np.ndarray, label: str) -> Dict[str, Any]:
        ex = SongFeatureExtractor(Path(f"{label}.wav"), cfg, audio=(y.astype(np.float32), sr))
        return ex.extract(index=0)

    def key_detail(r: Dict[str, Any]) -> str:
        d = r.get("_diagnostics", {})
        return (f"got {r['Key']} (2nd {d.get('key_runner_up', '?')}, "
                f"conf {d.get('key_confidence', 0):.2f}, "
                f"tiebreak {d.get('relative_tiebreak')} margin {d.get('tiebreak_margin')}, "
                f"chords {d.get('labelled_beats')}/{d.get('total_beats')} "
                f"[{d.get('first_chord')}..{d.get('last_chord')}])")

    print(f"song_features.py version {VERSION}", flush=True)
    print("Synthesising test audio and extracting features...\n", flush=True)

    # 1. E major I-IV-V-I -- the exact case that was coming out as B major.
    e_major = [["E3", "G#3", "B3"], ["A3", "C#4", "E4"],
               ["B3", "D#4", "F#4"], ["E3", "G#3", "B3"]]
    r = run(_synth_progression(e_major, bpm=120, sr=sr), "e_major")
    check(r["Key"] == "E major", "key: E major progression", key_detail(r))
    check(abs(r["Tempo"] - 120) < 6, "tempo: 120 BPM click", f"got {r['Tempo']:.1f}")

    # 2. C major I-vi-IV-V-I, resolving to the tonic the way real songs do.
    c_major = [["C4", "E4", "G4"], ["A3", "C4", "E4"], ["F3", "A3", "C4"],
               ["G3", "B3", "D4"], ["C4", "E4", "G4"]]
    r = run(_synth_progression(c_major, bpm=90, sr=sr), "c_major")
    check(r["Key"] == "C major", "key: C major, resolving to tonic", key_detail(r))
    check(abs(r["Tempo"] - 90) < 6, "tempo: 90 BPM", f"got {r['Tempo']:.1f}")

    # 2b. The same four chords as an unresolved loop. Relative keys share every
    #     note AND every chord transition, so with no melody, no cadence and no
    #     ending, C major and A minor are genuinely indistinguishable here -- a
    #     musician given only this loop would also be guessing. The correct
    #     behaviour is therefore not to be right, but to be honest: pick one of
    #     the pair and report low confidence so the row can be filtered.
    loop = [["C4", "E4", "G4"], ["A3", "C4", "E4"],
            ["F3", "A3", "C4"], ["G3", "B3", "D4"]]
    r = run(_synth_progression(loop, bpm=90, sr=sr), "loop")
    conf = (r.get("_diagnostics", {}) or {}).get("key_confidence") or 0.0
    check(r["Key"] in ("C major", "A minor") and conf < 0.35,
          "key: unresolved loop flagged uncertain", key_detail(r))

    # 3. A minor i-VI-III-VII
    a_minor = [["A3", "C4", "E4"], ["F3", "A3", "C4"],
               ["C4", "E4", "G4"], ["G3", "B3", "D4"]]
    r = run(_synth_progression(a_minor, bpm=100, sr=sr), "a_minor")
    check(r["Key"] == "A minor", "key: A minor progression", key_detail(r))

    # 4. Brightness: a dark tone vs a bright one.
    t = np.arange(sr * 20) / sr
    dark = 0.4 * np.sin(2 * np.pi * 120 * t)
    bright = 0.4 * np.sin(2 * np.pi * 5000 * t)
    rd, rb = run(dark, "dark"), run(bright, "bright")
    check(rd["Spectral Brightness"] < 600, "brightness: 120 Hz tone reads dark",
          f"got {rd['Spectral Brightness']:.0f} Hz")
    check(rb["Spectral Brightness"] > 3000, "brightness: 5 kHz tone reads bright",
          f"got {rb['Spectral Brightness']:.0f} Hz")
    check(rb["Spectral Brightness"] > rd["Spectral Brightness"] * 3,
          "brightness: bright clearly beats dark", "ordering holds")

    # 5. Loudness ordering and dynamic range.
    quiet = 0.02 * np.sin(2 * np.pi * 440 * t)
    loud = 0.9 * np.sin(2 * np.pi * 440 * t)
    rq, rl = run(quiet, "quiet"), run(loud, "loud")
    check(rq["Loudness"] < rl["Loudness"] - 15, "loudness: quiet reads much quieter",
          f"{rq['Loudness']:.1f} vs {rl['Loudness']:.1f}")

    swell = 0.9 * np.sin(2 * np.pi * 440 * t) * np.abs(np.sin(2 * np.pi * 0.05 * t))
    rs = run(swell, "swell")
    check(rs["Dynamic Range"] > rl["Dynamic Range"] + 5,
          "dynamic range: swelling tone beats steady",
          f"{rs['Dynamic Range']:.1f} vs {rl['Dynamic Range']:.1f} dB")

    # 6. Distortion: a clean sine vs a hard-clipped one.
    clipped = np.clip(3.0 * np.sin(2 * np.pi * 220 * t), -1.0, 1.0)
    rc = run(clipped, "clipped")
    clean = run(0.5 * np.sin(2 * np.pi * 220 * t), "clean")
    check(rc["Distortion"] > clean["Distortion"] + 0.2, "distortion: clipped beats clean",
          f"{rc['Distortion']:.2f} vs {clean['Distortion']:.2f}")

    # 7. Melodic range: a steady tone should read flat, a sweep should not.
    flat = run(0.4 * np.sin(2 * np.pi * 440 * t), "flat")["Melodic Range"]
    sweep_hz = 220.0 * (2.0 ** (np.linspace(0, 2, t.size)))          # two octaves up
    swept = run(0.4 * np.sin(2 * np.pi * np.cumsum(sweep_hz) / sr), "sweep")["Melodic Range"]
    check(flat is None or flat < 2.0, "melodic range: steady tone reads flat",
          f"got {flat}")
    check(swept is not None and swept > 12.0, "melodic range: two-octave sweep reads wide",
          f"got {swept}")

    passed = sum(1 for ok, _, _ in results if ok)
    print(f"\n{passed}/{len(results)} checks passed  (song_features.py {VERSION})")
    if passed < len(results):
        print("Failures above are worth telling me about -- they point at which "
              "feature to fix.", file=sys.stderr)
    return 0 if passed == len(results) else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def collect_files(inputs: Iterable[str], exts: Optional[Iterable[str]] = None) -> List[Path]:
    """Every audio file under the given files/folders, searched recursively."""
    wanted = {("." + e.lower().lstrip(".")) for e in exts} if exts else set(AUDIO_EXTS)
    files: List[Path] = []
    seen: set = set()
    for item in inputs:
        p = Path(item).expanduser()
        if p.is_dir():
            found = [q for q in p.rglob("*")
                     if q.is_file() and q.suffix.lower() in wanted
                     and not q.name.startswith((".", "._"))]
        elif p.is_file():
            found = [p]
        else:
            print(f"[warn] no such path: {p}", file=sys.stderr)
            continue
        for q in sorted(found, key=lambda x: str(x).lower()):
            key = str(q.resolve()).lower()
            if key not in seen:
                seen.add(key)
                files.append(q)
    return files


def _worker(args: Tuple[Path, Config, int]) -> Dict[str, Any]:
    path, cfg, idx = args
    try:
        return extract_features(path, cfg, index=idx)
    except Exception as exc:
        return {"Index": idx, "Song Name": path.stem, "_file": str(path),
                "_error": f"{type(exc).__name__}: {exc}",
                "_traceback": traceback.format_exc(limit=3)}


def sources_path_for(out: Path) -> Path:
    return out.with_name(out.name + ".sources.txt")


def read_existing(out: Path) -> Tuple[int, set]:
    """Return (next free Index, set of song names already in the CSV)."""
    if not out.exists() or out.stat().st_size == 0:
        return 1, set()
    names, idxs = set(), []
    try:
        with out.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("Song Name"):
                    names.add(r["Song Name"])
                raw = str(r.get("Index", "")).strip()
                if raw.isdigit():
                    idxs.append(int(raw))
    except Exception:
        return 1, set()
    return (max(idxs) + 1 if idxs else len(names) + 1), names


def write_csv(rows: List[Dict[str, Any]], out: Path, append: bool = False) -> None:
    has_content = out.exists() and out.stat().st_size > 0
    mode = "a" if (append and has_content) else "w"
    try:
        with out.open(mode, newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
            if mode == "w":
                w.writeheader()
            for r in rows:
                w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in COLUMNS})
    except PermissionError:
        raise PermissionError(
            f"Could not write {out} -- the file is open in Excel. "
            f"Close it and run the command again."
        ) from None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Extract musical features from audio files with librosa.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("inputs", nargs="*", help="audio files and/or folders")
    ap.add_argument("-o", "--out", type=Path, help="write the table to this CSV")
    ap.add_argument("--json", type=Path, help="also dump every diagnostic to JSON")
    ap.add_argument("--self-test", action="store_true",
                    help="check the extractor against synthetic audio with known answers")
    ap.add_argument("--version", action="version", version=f"song_features.py {VERSION}")
    ap.add_argument("--sr", type=int, default=22050, help="analysis sample rate")
    ap.add_argument("--timbre-sr", type=int, default=44100,
                    help="sample rate for loudness/brightness/distortion")
    ap.add_argument("--hop", type=int, default=512, help="hop length in samples")
    ap.add_argument("--offset", type=float, default=0.0, help="skip N seconds at start")
    ap.add_argument("--duration", type=float, default=None, help="analyse only the first N seconds")
    ap.add_argument("--pitch", choices=["pyin", "piptrack", "none"], default="pyin",
                    help="melodic-range tracker: pyin is accurate but slow")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes")
    ap.add_argument("--quiet", action="store_true", help="hide per-stage progress")
    ap.add_argument("--ext", nargs="+", metavar="EXT",
                    help="only these file types, e.g. --ext m4a (default: all audio)")
    ap.add_argument("--overwrite", action="store_true",
                    help="start the CSV again from scratch (default: add to it)")
    ap.add_argument("--redo", action="store_true",
                    help="re-process songs already in the CSV instead of skipping them")
    ap.add_argument("--artist-from-folder", action="store_true",
                    help="use the containing folder's name as the Artist when tags are missing")
    ap.add_argument("-a", "--append", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--skip-done", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test()
    if not args.inputs:
        ap.error("give me some audio files or folders (or use --self-test)")

    append = not args.overwrite
    skip_done = not (args.overwrite or args.redo)

    cfg = Config(
        sr=args.sr, timbre_sr=args.timbre_sr, hop_length=args.hop,
        offset=args.offset, duration=args.duration, pitch=args.pitch,
        verbose=not args.quiet and args.workers <= 1,
        artist_from_folder=args.artist_from_folder,
    )

    files = collect_files(args.inputs, args.ext)
    if not files:
        print("No audio files found.", file=sys.stderr)
        return 1
    print(f"Found {len(files)} audio file(s).", file=sys.stderr)

    if args.out and args.overwrite:
        sp = sources_path_for(args.out)
        if sp.exists():
            sp.unlink()

    start_index, done_names = 1, set()
    done_paths: set = set()
    if args.out and append:
        start_index, done_names = read_existing(args.out)
        sp = sources_path_for(args.out)
        if sp.exists():
            done_paths = {ln.strip().lower()
                          for ln in sp.read_text(encoding="utf-8").splitlines() if ln.strip()}

    if args.out and skip_done:
        before = len(files)
        files = [p for p in files
                 if str(p.resolve()).lower() not in done_paths and p.stem not in done_names]
        if before - len(files):
            print(f"Skipping {before - len(files)} already done.", file=sys.stderr)
        if not files:
            print("Nothing new to do.", file=sys.stderr)
            return 0

    jobs = [(p, cfg, start_index + i) for i, p in enumerate(files)]
    rows: List[Dict[str, Any]] = []

    def label(p: Path) -> str:
        return f"{p.parent.name}/{p.name}" if p.parent.name else p.name

    if args.workers > 1 and len(jobs) > 1:
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, row in enumerate(pool.map(_worker, jobs), 1):
                rows.append(row)
                print(f"[{i}/{len(jobs)}] {label(Path(row['_file']))}", file=sys.stderr)
    else:
        for i, job in enumerate(jobs, 1):
            print(f"[{i}/{len(jobs)}] {label(job[0])}", file=sys.stderr)
            rows.append(_worker(job))

    rows.sort(key=lambda r: r.get("Index") or 0)
    failed = [r for r in rows if "_error" in r]
    for r in failed:
        print(f"[error] {r['_file']}: {r['_error']}", file=sys.stderr)

    if args.out:
        try:
            write_csv(rows, args.out, append=append)
        except PermissionError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 3
        ok_paths = [r["_file"] for r in rows if "_error" not in r]
        if ok_paths:
            with sources_path_for(args.out).open("a", encoding="utf-8") as fh:
                for pth in ok_paths:
                    fh.write(str(Path(pth).resolve()) + "\n")
        msg = f"{'Added' if append else 'Wrote'} {len(rows) - len(failed)} row(s) to {args.out}"
        if failed:
            msg += f"; {len(failed)} failed (see errors above)"
        print(msg, file=sys.stderr)
    else:
        for r in rows:
            print(json.dumps({c: r.get(c) for c in COLUMNS}, indent=2, default=str))

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        print(f"Wrote diagnostics to {args.json}", file=sys.stderr)

    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
