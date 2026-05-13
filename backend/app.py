import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify
from flask_cors import CORS
from config.settings import FLASK_PORT, FLASK_DEBUG
from models.model_utils import get_db, load_best_model
from backend.routes.predict import predict_bp
from backend.routes.history import history_bp
from backend.routes.health import health_bp


# Initialize Flask app

app = Flask(__name__)
CORS(app)


# Load model and db once on startup and attach to app context

def initialize_app(app):
    db = get_db()
    artifact, model_doc = load_best_model(db)
    app.config["DB"] = db
    app.config["ARTIFACT"] = artifact
    app.config["MODEL_DOC"] = model_doc
    print(f"Model loaded: {model_doc['model_type']} | RMSE: {model_doc['metrics']['rmse']}")


# Register blueprints

app.register_blueprint(predict_bp)
app.register_blueprint(history_bp)
app.register_blueprint(health_bp)


# Index route

@app.route("/")
def index():
    return jsonify({
        "api": "AQI Predictor",
        "version": "1.0",
        "endpoints": ["/predict", "/history", "/health"]
    })


if __name__ == "__main__":
    initialize_app(app)
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=FLASK_DEBUG)