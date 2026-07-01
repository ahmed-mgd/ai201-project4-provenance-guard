"""Detection pipeline — runs the signals and combines them into one score.

The combined score is a fixed weighted average of each signal's p_ai (see
planning.md → "Combining signals into a confidence score"). Weights and
thresholds live here as documented constants and are echoed in the API response.
"""

from __future__ import annotations

from .signals import lexical_signal, llm_signal, stylometry_signal

# Ensemble weights (sum to 1.0). LLM highest (only meaning-aware signal),
# lexical lowest (most evadable). Matches planning.md.
WEIGHTS = {"llm": 0.50, "stylometry": 0.30, "lexical": 0.20}

# Below this word count we treat the text as thin evidence and shrink the
# combined score toward 0.5 (planning.md → edge case: very short submission).
# This pulls short text toward Uncertain so a creator is not accused on a
# sentence or two. SHRINK = 0.5 halves the distance from 0.5.
SHORT_TEXT_WORDS = 25
SHORT_TEXT_SHRINK = 0.5

# Attribution thresholds on the combined p_ai. Calibrated in M4 against the
# reference inputs: a weighted average of three noisy signals compresses toward
# the middle, so real AI text tops out around 0.6-0.7, not 0.9. The separation
# point between AI and human writing sits near 0.55, so that is the AI bar. The
# human bar stays lower (0.35), which keeps the band asymmetric: ambiguous work
# falls to "uncertain" instead of being accused of AI. See planning.md →
# Uncertainty Representation.
AI_THRESHOLD = 0.55       # p_ai >= this  -> likely_ai
HUMAN_THRESHOLD = 0.35    # p_ai <= this  -> likely_human


def classify(text: str) -> dict:
    """Run all signals and return the combined attribution result."""
    signals = {
        "llm": llm_signal.analyze(text),
        "stylometry": stylometry_signal.analyze(text),
        "lexical": lexical_signal.analyze(text),
    }

    weights = WEIGHTS
    probability_ai = sum(weights[name] * signals[name]["p_ai"] for name in weights)

    # Thin evidence: shrink toward 0.5 so short text can't produce a confident
    # accusation. A clearly-human short note can still land human; a short AI-ish
    # note gets pulled back to Uncertain.
    word_count = len((text or "").split())
    short_text = word_count < SHORT_TEXT_WORDS
    if short_text:
        probability_ai = 0.5 + (probability_ai - 0.5) * SHORT_TEXT_SHRINK

    probability_ai = round(_clamp(probability_ai), 3)
    confidence = round(max(probability_ai, 1.0 - probability_ai), 3)
    attribution = attribution_from_score(probability_ai)

    return {
        "attribution": attribution,
        "probability_ai": probability_ai,
        "confidence": confidence,
        "weights": weights,
        "short_text": short_text,
        "signals": signals,
    }


def attribution_from_score(probability_ai: float) -> str:
    if probability_ai >= AI_THRESHOLD:
        return "likely_ai"
    if probability_ai <= HUMAN_THRESHOLD:
        return "likely_human"
    return "uncertain"


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
