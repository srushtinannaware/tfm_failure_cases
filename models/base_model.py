"""
Abstract base class for all models.
"""

class BaseModel:

    def fit(self, X, y):
        """Train model."""
        pass

    def predict(self, X):
        """Predict."""
        pass

    def predict_proba(self, X):
        """Predict probabilities."""
        pass
  