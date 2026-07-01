"""Provenance Guard — Flask API (Milestone 4).

Scope so far:
  POST /submit   classify text with the multi-signal ensemble, return a result
  GET  /log      return recent audit-log entries as JSON
  GET  /health   liveness check

The label is still a placeholder here. Milestone 5 adds the real transparency
label, the appeal endpoint, and rate limiting.
"""

from __future__ import annotations

import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from provenance_guard import pipeline, store

load_dotenv()

app = Flask(__name__)
store.init_db()


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

    # Run the multi-signal ensemble: combined probability_ai, confidence,
    # attribution, and each signal's individual score.
    result = pipeline.classify(text)
    signals = result["signals"]

    # Label is still a placeholder; the real transparency label lands in M5.
    label = {
        "variant": "placeholder",
        "text": f"[placeholder] attribution={result['attribution']} "
                f"(real label added in Milestone 5)",
    }

    content_id = str(uuid.uuid4())
    created_at = store.now_iso()

    store.save_submission({
        "content_id": content_id,
        "text": text,
        "creator_id": creator_id,
        "attribution": result["attribution"],
        "probability_ai": result["probability_ai"],
        "confidence": result["confidence"],
        "llm_score": signals["llm"]["p_ai"],
        "stylometry_score": signals["stylometry"]["p_ai"],
        "lexical_score": signals["lexical"]["p_ai"],
        "status": "classified",
        "created_at": created_at,
    })

    # Audit log records each signal's individual score alongside the combined
    # confidence and probability.
    store.append_audit({
        "content_id": content_id,
        "creator_id": creator_id,
        "timestamp": created_at,
        "attribution": result["attribution"],
        "probability_ai": result["probability_ai"],
        "confidence": result["confidence"],
        "signals": {
            "llm": signals["llm"]["p_ai"],
            "stylometry": signals["stylometry"]["p_ai"],
            "lexical": signals["lexical"]["p_ai"],
        },
        "weights": result["weights"],
        "status": "classified",
    })

    return jsonify({
        "content_id": content_id,
        "attribution": result["attribution"],
        "confidence": result["confidence"],
        "probability_ai": result["probability_ai"],
        "short_text": result["short_text"],
        "weights": result["weights"],
        "signals": {
            name: {"p_ai": s["p_ai"], "features": s["features"]}
            for name, s in signals.items()
        },
        "label": label,
    })


@app.get("/log")
def log():
    return jsonify({"entries": store.read_audit_log()})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
