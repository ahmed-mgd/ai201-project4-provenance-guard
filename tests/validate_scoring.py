"""Validation script for the detection pipeline (Milestone 4).

Runs a set of deliberately chosen inputs through the pipeline and prints each
signal's individual score next to the combined score, so we can see whether the
scores separate the way we expect. Run from the repo root:

    python -m tests.validate_scoring

This is the "reference inputs" test described in planning.md → Uncertainty
Representation.
"""

from __future__ import annotations

from dotenv import load_dotenv

from provenance_guard import pipeline
from provenance_guard.signals import lexical_signal, stylometry_signal

load_dotenv()

CASES = {
    "clearly AI-generated": (
        "Artificial intelligence represents a transformative paradigm shift in "
        "modern society. It is important to note that while the benefits of AI "
        "are numerous, it is equally essential to consider the ethical "
        "implications. Furthermore, stakeholders across various sectors must "
        "collaborate to ensure responsible deployment."
    ),
    "clearly human-written": (
        "ok so i finally tried that new ramen place downtown and honestly? "
        "underwhelming. the broth was fine but they put WAY too much sodium in "
        "it and i was thirsty for like three hours after. my friend got the "
        "spicy version and said it was better. probably won't go back unless "
        "someone drags me there"
    ),
    "borderline: formal human": (
        "The relationship between monetary policy and asset price inflation has "
        "been extensively studied in the literature. Central banks face a "
        "fundamental tension between their mandate for price stability and the "
        "unintended consequences of prolonged low interest rates on equity and "
        "real estate valuations."
    ),
    "borderline: lightly edited AI": (
        "I've been thinking a lot about remote work lately. There are genuine "
        "tradeoffs — flexibility and no commute on one side, isolation and "
        "blurred work-life boundaries on the other. Studies show productivity "
        "varies widely by individual and role type."
    ),
}


def main() -> None:
    print("=== Signals 2 & 3 standalone (no LLM call) ===")
    for name, text in CASES.items():
        s = stylometry_signal.analyze(text)["p_ai"]
        l = lexical_signal.analyze(text)["p_ai"]
        print(f"  {name:32s} stylometry={s:<5} lexical={l}")

    print("\n=== Full ensemble ===")
    for name, text in CASES.items():
        r = pipeline.classify(text)
        sig = {k: v["p_ai"] for k, v in r["signals"].items()}
        print(
            f"  {name:32s} llm={sig['llm']:<5} "
            f"stylo={sig['stylometry']:<5} lex={sig['lexical']:<5} "
            f"=> p_ai={r['probability_ai']:<5} conf={r['confidence']:<5} "
            f"[{r['attribution']}]"
        )


if __name__ == "__main__":
    main()
