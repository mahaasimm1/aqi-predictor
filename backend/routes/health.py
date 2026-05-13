from flask import Blueprint, jsonify, current_app
from datetime import datetime, timezone

health_bp = Blueprint("health", __name__)

@health_bp.route("/health")
def health():
    model_doc = current_app.config.get("MODEL_DOC", {})
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_type": model_doc.get("model_type", "unknown"),
        "model_trained_at": model_doc.get("trained_at", "unknown").isoformat()
        if model_doc.get("trained_at") else "unknown"
    })