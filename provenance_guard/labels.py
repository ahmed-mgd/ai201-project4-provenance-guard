"""Transparency label builder.

Maps an ensemble result to one of three reader-facing label variants. The label
text changes with the verdict, not just the number. The exact copy and the
thresholds match planning.md → Transparency Label Variants.
"""

from __future__ import annotations

from .pipeline import AI_THRESHOLD, HUMAN_THRESHOLD


def build_label(probability_ai: float, confidence: float) -> dict:
    """Return {'variant': str, 'text': str} for the given result."""
    pct = round(confidence * 100)

    if probability_ai >= AI_THRESHOLD:
        variant = "high_confidence_ai"
        text = (
            f"Likely AI-generated. Our checks lean toward this piece being "
            f"written with generative AI (confidence: {pct}%). This is an "
            f"automated estimate, not a certainty — the creator can appeal if "
            f"they believe it's wrong."
        )
    elif probability_ai <= HUMAN_THRESHOLD:
        variant = "high_confidence_human"
        text = (
            f"Likely human-written. Our checks found no strong signs of AI "
            f"generation (confidence: {pct}%). This is an automated estimate and "
            f"not a guarantee of authorship."
        )
    else:
        variant = "uncertain"
        text = (
            f"Not enough signal to tell. Our checks were mixed and we can't "
            f"confidently say whether this was written by a person or AI "
            f"(confidence: {pct}%). We're showing this openly rather than "
            f"guessing."
        )

    return {"variant": variant, "text": text}
