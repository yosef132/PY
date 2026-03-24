"""
ML Signal Scorer Model
XGBoost classifier that predicts whether a signal will be profitable.
This is the AI brain that learns over time.
"""

import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from src.utils.config import PROJECT_ROOT
from src.utils.logger import setup_logger

logger = setup_logger("ml_model")


class MLSignalScorer:
    """
    XGBoost model that scores trading signals.
    Learns from historical trade outcomes to predict which signals will profit.
    """

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.is_trained = False
        self.model_dir = PROJECT_ROOT / "data" / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list) -> dict:
        """
        Train the XGBoost model on historical trade data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (1 = win, 0 = loss)
            feature_names: List of feature names

        Returns:
            Dict with training metrics
        """
        self.feature_names = feature_names

        n_samples = len(X)
        n_features = X.shape[1] if len(X) > 0 else 1

        if n_samples < 20:
            logger.warning(f"  Only {n_samples} samples - too few for reliable training")
            logger.warning(f"  Need at least 50+ trades for meaningful ML learning")

        # Adaptive regularization: scale complexity DOWN when data is scarce.
        # With few samples and many features the model easily memorises labels.
        # Rule of thumb: need ~10-20 samples per feature for generalisation.
        ratio = n_samples / max(n_features, 1)

        if ratio < 5:           # very few samples per feature → conservative but still learning
            max_depth        = 2
            n_estimators     = 50   # more trees needed to find weak patterns
            min_child_weight = 5
            reg_alpha        = 0.5
            reg_lambda       = 2.0
            subsample        = 0.7
            colsample_bytree = 0.7
            logger.warning(f"  Low data ratio ({ratio:.1f} samples/feature) — using conservative model")
        elif ratio < 10:        # moderate scarcity
            max_depth        = 3
            n_estimators     = 50
            min_child_weight = 7
            reg_alpha        = 0.5
            reg_lambda       = 3.0
            subsample        = 0.7
            colsample_bytree = 0.7
        else:                   # enough data — normal settings
            max_depth        = 4
            n_estimators     = 100
            min_child_weight = 3
            reg_alpha        = 0.1
            reg_lambda       = 1.0
            subsample        = 0.8
            colsample_bytree = 0.8

        logger.info(f"  XGBoost config: depth={max_depth}, trees={n_estimators}, "
                    f"samples/feature ratio={ratio:.1f}")

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,       # lower LR = more robust, works with any tree count
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
        )

        # Time-series cross-validation (respects temporal order)
        n_splits = min(5, max(2, len(X) // 20))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        try:
            cv_scores = cross_val_score(self.model, X, y, cv=tscv, scoring="accuracy")
        except Exception as e:
            logger.warning(f"  CV failed: {e}, training on full data")
            cv_scores = np.array([0.5])

        # Train on full dataset
        self.model.fit(X, y)
        self.is_trained = True

        # Get predictions on training data (for analysis)
        y_pred = self.model.predict(X)
        train_acc = accuracy_score(y, y_pred)

        # Feature importance
        importances = self.model.feature_importances_
        top_features = sorted(zip(feature_names, importances),
                              key=lambda x: x[1], reverse=True)[:10]

        metrics = {
            "cv_accuracy": round(cv_scores.mean() * 100, 1),
            "cv_std": round(cv_scores.std() * 100, 1),
            "train_accuracy": round(train_acc * 100, 1),
            "n_samples": len(X),
            "n_features": len(feature_names),
            "win_rate_in_data": round(y.mean() * 100, 1),
            "top_features": top_features,
        }

        logger.info(f"  ML Model trained:")
        logger.info(f"    CV Accuracy: {metrics['cv_accuracy']}% (+/- {metrics['cv_std']}%)")
        logger.info(f"    Train Accuracy: {metrics['train_accuracy']}%")
        logger.info(f"    Top features:")
        for feat, imp in top_features[:5]:
            logger.info(f"      {feat}: {imp:.3f}")

        return metrics

    def predict(self, features: dict) -> tuple:
        """
        Predict whether a signal will be profitable.

        Args:
            features: Dict of feature_name -> value

        Returns:
            (prediction: 0 or 1, probability: float 0.0 to 1.0)
        """
        if not self.is_trained:
            return 1, 0.5  # Default: allow trade with neutral confidence

        # Build feature vector in correct order
        X = np.array([[features.get(name, 0.0) for name in self.feature_names]])

        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0][1]  # P(win)

        return int(prediction), float(probability)

    def score_signal(self, signal, df_row, feature_builder) -> float:
        """
        Score a signal using the ML model.

        Returns:
            ML confidence score (0.0 to 1.0)
        """
        if not self.is_trained:
            return signal.confidence  # Pass through original confidence

        features = feature_builder.extract_features(signal, df_row)
        _, ml_probability = self.predict(features)

        # Blend ML score with strategy confidence
        # ML has 60% weight, strategy has 40% weight
        blended = ml_probability * 0.6 + signal.confidence * 0.4

        return round(blended, 3)

    def save(self, filename: str = "xgb_signal_scorer.pkl"):
        """Save trained model to disk."""
        if not self.is_trained:
            logger.warning("  Cannot save: model not trained")
            return

        filepath = self.model_dir / filename
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
        }, filepath)
        logger.info(f"  Model saved to {filepath}")

    def load(self, filename: str = "xgb_signal_scorer.pkl") -> bool:
        """Load model from disk."""
        filepath = self.model_dir / filename
        if not filepath.exists():
            logger.warning(f"  No saved model found at {filepath}")
            return False

        data = joblib.load(filepath)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.is_trained = True
        logger.info(f"  Model loaded from {filepath}")
        return True
