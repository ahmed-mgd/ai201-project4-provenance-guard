"""Signal 3 — Lexical regularity and AI "tells" (pure Python).

Surface fingerprints of instruction-tuned model output:
  - how often stock AI phrases show up ("delve," "moreover," "it's important
    to note," ...)
  - punctuation density, mainly em-dash overuse
  - back-to-back repeated bigrams

This catches the phrasing signature of AI text, a different failure mode than
the semantic check (Signal 1) or the structural check (Signal 2). It is easy to
evade with paraphrase and can false-positive on formal or ESL human writing.
See planning.md → Detection Signals → Signal 3.
"""

from __future__ import annotations

import re

# Stock phrasing that instruction-tuned models overuse.
AI_TELLS = [
    "it's important to note",
    "it is important to note",
    "in today's world",
    "in the world of",
    "delve into",
    "delve",
    "tapestry",
    "navigate the complexities",
    "navigating the complexities",
    "a testament to",
    "when it comes to",
    "in conclusion",
    "moreover",
    "furthermore",
    "additionally",
    "on the other hand",
    "plays a crucial role",
    "plays a vital role",
    "paradigm shift",
    "ever-evolving",
    "unlock the",
    "harness the power",
]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def analyze(text: str) -> dict:
    """Return {'p_ai': float, 'features': {...}} for the lexical signal."""
    lower = text.lower()
    words = _words(text)
    n = len(words) or 1

    tell_hits = sum(lower.count(phrase) for phrase in AI_TELLS)
    tell_rate = tell_hits / (n / 100)  # tells per 100 words

    em_dashes = text.count("—") + text.count(" - ")
    em_dash_rate = em_dashes / (n / 100)

    # Back-to-back repeated bigrams (AI text often loops phrasing).
    bigrams = list(zip(words, words[1:]))
    repeat_bigrams = sum(
        1 for i in range(1, len(bigrams)) if bigrams[i] == bigrams[i - 1]
    )
    repeat_rate = repeat_bigrams / max(1, len(bigrams))

    # Map each component to [0,1], then take a weighted average. Tells carry the
    # most weight because they are the strongest lexical signature.
    tells_ai = _clamp(tell_rate / 2.0)     # ~2 tells / 100 words -> strong
    dash_ai = _clamp(em_dash_rate / 3.0)   # heavy em-dash use -> AI-ish
    repeat_ai = _clamp(repeat_rate / 0.05)
    p_ai = _clamp(0.6 * tells_ai + 0.2 * dash_ai + 0.2 * repeat_ai)

    return {
        "p_ai": round(p_ai, 3),
        "features": {
            "ai_tell_hits": tell_hits,
            "ai_tells_per_100w": round(tell_rate, 2),
            "em_dash_per_100w": round(em_dash_rate, 2),
            "repeat_bigram_rate": round(repeat_rate, 3),
            "word_count": n,
        },
    }


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
