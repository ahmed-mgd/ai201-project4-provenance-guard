"""Signal 2 — Stylometric variability (pure Python, no external libraries).

Measures two structural properties of the text:
  - burstiness: coefficient of variation of sentence length (stdev / mean)
  - vocabulary diversity: type-token ratio (unique words / total words)

Humans mix long and short sentences and use varied words, so their writing has
high burstiness and a high type-token ratio. AI writing trends uniform and
repetitive, so both numbers drop. Returns a probability that the text is AI.

See planning.md → Detection Signals → Signal 2 for what this misses.
"""

from __future__ import annotations

import re
from statistics import mean, pstdev


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def analyze(text: str) -> dict:
    """Return {'p_ai': float, 'features': {...}} for the stylometry signal."""
    sentences = _split_sentences(text)
    words = _words(text)

    # Too little text to measure structure reliably. Stay neutral.
    if len(sentences) < 2 or len(words) < 15:
        return {
            "p_ai": 0.5,
            "features": {
                "note": "text too short for stable stylometry",
                "sentence_count": len(sentences),
                "word_count": len(words),
            },
        }

    lengths = [len(_words(s)) for s in sentences]
    m = mean(lengths) or 1.0
    cv = pstdev(lengths) / m           # burstiness
    ttr = len(set(words)) / len(words)  # type-token ratio

    # Map each feature to a "how AI-like" value in [0,1], then average them.
    #   low burstiness  -> more AI   (human prose is often cv ~0.5+, AI < 0.35)
    #   low diversity   -> more AI   (higher ttr = more human)
    burst_ai = _clamp(1.0 - (cv / 0.6))     # cv 0 -> 1.0 ; cv 0.6 -> 0.0
    ttr_ai = _clamp((0.75 - ttr) / 0.35)    # ttr 0.75 -> 0 ; ttr 0.40 -> 1.0
    p_ai = _clamp(0.5 * burst_ai + 0.5 * ttr_ai)

    return {
        "p_ai": round(p_ai, 3),
        "features": {
            "sentence_count": len(sentences),
            "word_count": len(words),
            "mean_sentence_len": round(m, 2),
            "sentence_len_cv": round(cv, 3),
            "type_token_ratio": round(ttr, 3),
        },
    }


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
