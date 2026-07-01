from __future__ import annotations

import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from provenance_guard import labels, pipeline, store

load_dotenv()

app = Flask(__name__)
store.init_db()

# Rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/submit")
@limiter.limit("10 per minute;100 per day")
def submit():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    creator_id = data.get("creator_id") or "unknown"

    if not text:
        return jsonify({"error": "Field 'text' is required and cannot be empty."}), 400

    # Run the multi-signal ensemble.
    result = pipeline.classify(text)
    signals = result["signals"]

    # Real transparency label, chosen by the confidence/verdict.
    label = labels.build_label(result["probability_ai"], result["confidence"])

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

    store.append_audit({
        "event": "classification",
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
        "label_variant": label["variant"],
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


@app.post("/appeal")
@limiter.limit("20 per hour")
def appeal():
    data = request.get_json(silent=True) or {}
    content_id = data.get("content_id")
    creator_reasoning = (data.get("creator_reasoning") or "").strip()

    if not content_id:
        return jsonify({"error": "Field 'content_id' is required."}), 400
    if not creator_reasoning:
        return jsonify({"error": "Field 'creator_reasoning' is required and cannot be empty."}), 400

    submission = store.get_submission(content_id)
    if submission is None:
        return jsonify({"error": f"No submission with content_id {content_id}."}), 404

    updated = store.save_appeal(content_id, creator_reasoning)
    return jsonify({
        "content_id": content_id,
        "status": updated["status"],
        "message": "Appeal received. This submission is now under review.",
        "appeal_reasoning": creator_reasoning,
        "original_attribution": submission["attribution"],
    })


@app.get("/log")
def log():
    return jsonify({"entries": store.read_audit_log()})


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Rate limit exceeded. Please slow down and try again later.",
        "detail": str(e.description),
    }), 429


if __name__ == "__main__":
    app.run(debug=True, port=5000)
