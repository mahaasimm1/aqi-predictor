import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.model_utils import load_best_model, load_latest_features, predict, get_db


def get_model_and_features(db=None):
    if db is None:
        db = get_db()
    artifact, model_doc = load_best_model(db)
    feature_cols = model_doc["feature_cols"]
    X, df = load_latest_features(feature_cols, db=db)
    return artifact, model_doc, X, df, feature_cols