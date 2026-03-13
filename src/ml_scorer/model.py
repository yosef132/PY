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

        if len(X) < 20:
            logger.warning(f"  Only {len(X)} samples - too few for reliable training")
            logger.warning(f"  Need at least 50+ trades for meaningful ML learning")

        # XGBoost with conservative settings to avoid overfitting
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=4,              # Shallow trees = less overfitting
            learning_rate=0.1,
            subsample=0.8,            # Use 80% of data per tree
            colsample_bytree=0.8,     # Use 80% of features per tree
            min_child_weight=3,       # Minimum samples in leaf
            reg_alpha=0.1,            # L1 regularization
            reg_lambda=1.0,           # L2 regularization
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
