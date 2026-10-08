"""
Loads the automatically-selected BEST model (from models/results/best_model.json)
and the auxiliary discourse_type classifier (if it was trained), and exposes a
single, simple function to predict on new text:

    predictor = load_predictor()
    result = predictor.predict(some_text)

This module ONLY does prediction/model-loading. It deliberately knows nothing
about Socratic feedback - see socratic_feedback.py for that (kept separate on
purpose, as required).

Used by: app/app.py (the Streamlit app). Can also be run directly for a quick
command-line check:
    python src/prediction.py "Some essay text to classify"
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import common


class PredictionResult:
    def __init__(self, effectiveness: str, effectiveness_probs: Optional[Dict[str, float]],
                 discourse_type: Optional[str], discourse_type_probs: Optional[Dict[str, float]]):
        self.effectiveness = effectiveness
        self.effectiveness_probs = effectiveness_probs
        self.discourse_type = discourse_type
        self.discourse_type_probs = discourse_type_probs


# =====================================================================================
# LOADERS FOR EACH MODEL TYPE (dispatch based on best_model.json's "model_type")
# =====================================================================================
class _SklearnTfidfPredictor:
    def __init__(self, model_key: str):
        import joblib

        self.vectorizer = joblib.load(config.model_path(model_key, "vectorizer.joblib"))
        self.model = joblib.load(config.model_path(model_key, "model.joblib"))
        self.labels = common.load_label_classes()
        self.id_to_label = {i: c for i, c in enumerate(self.labels)}

    def predict(self, text: str):
        X = self.vectorizer.transform([text])
        pred_id = self.model.predict(X)[0]
        label = self.id_to_label.get(pred_id, str(pred_id))
        probs = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            probs = {self.id_to_label.get(i, str(i)): float(p) for i, p in enumerate(proba)}
        return label, probs


class _TorchLSTMPredictor:
    def __init__(self, model_key: str):
        from _torch_lstm_shared import require_torch, LSTMClassifier

        torch, _ = require_torch()
        self.torch = torch
        cfg = common.load_json(config.model_path(model_key, "vocab_and_config.json"))
        self.vocab = cfg["vocab"]
        self.bidirectional = cfg["bidirectional"]
        self.labels = common.load_label_classes()
        self.id_to_label = {i: c for i, c in enumerate(self.labels)}

        wrapper = LSTMClassifier(len(self.vocab), len(self.labels), bidirectional=self.bidirectional)
        state = torch.load(config.model_path(model_key, "model_state.pt"), map_location="cpu")
        wrapper.net.load_state_dict(state)
        wrapper.net.eval()
        self.net = wrapper.net

    def predict(self, text: str):
        from data_preprocessing import clean_text

        cleaned = clean_text(text)
        encoded = common.encode_texts([cleaned], self.vocab, config.MAX_SEQUENCE_LENGTH)
        with self.torch.no_grad():
            logits = self.net(self.torch.tensor(encoded, dtype=self.torch.long))
            probs_tensor = self.torch.softmax(logits, dim=1)[0]
        pred_id = int(probs_tensor.argmax().item())
        label = self.id_to_label.get(pred_id, str(pred_id))
        probs = {self.id_to_label.get(i, str(i)): float(p) for i, p in enumerate(probs_tensor.tolist())}
        return label, probs


class _SentenceTransformerMLPPredictor:
    def __init__(self, model_key: str):
        import joblib
        from train_model_6 import require_sentence_transformers

        SentenceTransformer = require_sentence_transformers()
        cfg = common.load_json(config.model_path(model_key, "embedding_config.json"))
        self.embedder = SentenceTransformer(cfg["sentence_transformer_model"])
        self.mlp = joblib.load(config.model_path(model_key, "mlp_classifier.joblib"))
        self.labels = common.load_label_classes()
        self.id_to_label = {i: c for i, c in enumerate(self.labels)}

    def predict(self, text: str):
        embedding = self.embedder.encode([text])
        pred_id = self.mlp.predict(embedding)[0]
        label = self.id_to_label.get(pred_id, str(pred_id))
        probs = None
        if hasattr(self.mlp, "predict_proba"):
            proba = self.mlp.predict_proba(embedding)[0]
            probs = {self.id_to_label.get(i, str(i)): float(p) for i, p in enumerate(proba)}
        return label, probs


_LOADER_BY_TYPE = {
    "sklearn_tfidf": _SklearnTfidfPredictor,
    "torch_lstm": _TorchLSTMPredictor,
    "sentence_transformer_mlp": _SentenceTransformerMLPPredictor,
}


class _AuxiliaryTypePredictor:
    """Loads the auxiliary discourse_type classifier, if it was trained."""

    def __init__(self):
        import joblib

        model_key = "aux_discourse_type"
        vec_path = config.model_path(model_key, "vectorizer.joblib")
        model_path_ = config.model_path(model_key, "model.joblib")
        if not vec_path.is_file() or not model_path_.is_file():
            self.available = False
            return
        self.available = True
        self.vectorizer = joblib.load(vec_path)
        self.model = joblib.load(model_path_)

    def predict(self, text: str):
        if not self.available:
            return None, None
        X = self.vectorizer.transform([text])
        label = self.model.predict(X)[0]
        probs = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            probs = {c: float(p) for c, p in zip(self.model.classes_, proba)}
        return label, probs


class Predictor:
    """The single object app.py needs: predictor.predict(text) -> PredictionResult."""

    def __init__(self):
        if not config.BEST_MODEL_JSON.is_file():
            raise FileNotFoundError(
                "models/results/best_model.json was not found. Run the full pipeline first "
                "(python run_all.py), which trains all models and automatically selects the best one."
            )
        self.best_info = common.load_json(config.BEST_MODEL_JSON)
        model_type = self.best_info["model_type"]
        if model_type not in _LOADER_BY_TYPE:
            raise ValueError(f"Unknown model_type '{model_type}' in best_model.json.")

        print(f"[INFO] Loading best model: {self.best_info['best_model']} "
              f"(Macro F1 = {self.best_info['score']})")
        self.effectiveness_model = _LOADER_BY_TYPE[model_type](self.best_info["model_key"])
        self.type_model = _AuxiliaryTypePredictor()

    def predict(self, text: str) -> PredictionResult:
        if not text or not text.strip():
            raise ValueError("Please provide some text to analyze.")
        eff_label, eff_probs = self.effectiveness_model.predict(text)
        type_label, type_probs = self.type_model.predict(text)
        return PredictionResult(eff_label, eff_probs, type_label, type_probs)


def load_predictor() -> Predictor:
    return Predictor()


if __name__ == "__main__":
    sample_text = " ".join(sys.argv[1:]) or "I believe that schools should start later because students need more sleep."
    predictor = load_predictor()
    result = predictor.predict(sample_text)
    print(f"\nText: {sample_text}")
    print(f"Predicted effectiveness: {result.effectiveness}")
    if result.effectiveness_probs:
        print(f"Probabilities: {result.effectiveness_probs}")
    print(f"Predicted discourse type: {result.discourse_type}")
