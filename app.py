from __future__ import annotations

import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from provenance_guard import store
from provenance_guard.signals import llm_signal

load_dotenv()

app = Flask(__name__)
store.init_db()

# Attribution thresholds on a probability-of-AI score. Asymmetric on purpose:
# the bar to call something AI is higher than the bar to call it human (see
# planning.md → Uncertainty Representation). For now the score comes from signal
# 1 alone; Milestone 4 moves this onto the ensemble.
AI_THRESHOLD = 0.80
HUMAN_THRESHOLD = 0.35


def attribution_from_score(p_ai: float) -> str:
    if p_ai >= AI_THRESHOLD:
        return "likely_ai"
    if p_ai <= HUMAN_THRESHOLD:
        return "likely_human"
    return "uncertain"


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/submit")
def submit():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    creator_id = data.get("creator_id") or "unknown"

    if not text:
        return jsonify({"error": "Field 'text' is required and cannot be empty."}), 400

    # Signal 1: LLM semantic classifier. Returns {p_ai, features}.
    signal1 = llm_signal.analyze(text)
    llm_score = signal1["p_ai"]

    attribution = attribution_from_score(llm_score)

    # Placeholder confidence and label. Real ensemble confidence lands in M4 and
    # the real transparency label in M5.
    confidence = round(max(llm_score, 1.0 - llm_score), 3)
    label = {
        "variant": "placeholder",
        "text": f"[placeholder] attribution={attribution} "
                f"(real label added in Milestone 5)",
    }

    content_id = str(uuid.uuid4())
    created_at = store.now_iso()

    store.save_submission({
        "content_id": content_id,
        "text": text,
        "creator_id": creator_id,
        "attribution": attribution,
        "confidence": confidence,
        "llm_score": llm_score,
        "status": "classified",
        "created_at": created_at,
    })

    store.append_audit({
        "content_id": content_id,
        "creator_id": creator_id,
        "timestamp": created_at,
        "attribution": attribution,
        "confidence": confidence,
        "llm_score": llm_score,
        "status": "classified",
    })

    return jsonify({
        "content_id": content_id,
        "attribution": attribution,
        "confidence": confidence,
        "llm_score": llm_score,
        "signal_1": {"name": "llm_semantic", "p_ai": llm_score,
                     "features": signal1["features"]},
        "label": label,
    })


@app.get("/log")
def log():
    return jsonify({"entries": store.read_audit_log()})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
