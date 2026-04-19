from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
from pathlib import Path

class LogisticRegressionBaseline:
    """
    Logistic Regression baseline.
    Works on flattened MFCC features (mfcc_lr format from utils.get_datasets).
    Uses sklearn, so no GPU needed.
    """

    def __init__(self, max_iter: int = 1000, C: float = 1.0, solver: str = "saga", n_jobs: int = -1, random_state: int = 42):
        """
        Initializes the Logistic Regression baseline model.
        Args:
            max_iter: Maximum number of iterations for the logistic regression solver.
            C: Inverse of regularization strength; must be a positive float. Smaller values specify stronger regularization.
            solver: Algorithm to use in the optimization problem. "saga" is good for large datasets and supports L1 regularization.
            n_jobs: Number of CPU cores to use for computation. -1 means using all processors.
            random_state: Controls the randomness of the estimator. Pass an int for reproducible output across multiple function calls.
        """
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=max_iter, C=C, solver=solver, n_jobs=n_jobs, multi_class="multinomial", random_state=random_state)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fits the logistic regression model to the training data.

        Args:
            X_train: Training features, expected to be a 2D array of shape (n_samples, n_features).
            y_train: Training labels, expected to be a 1D array of shape (n_samples,).
        """
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for the input features.

        Args:
            X: Input features, expected to be a 2D array of shape (n_samples, n_features).
       
        Returns:
            Predicted class labels, a 1D array of shape (n_samples,).
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities for the input features.

        Args:
            X: Input features, expected to be a 2D array of shape (n_samples, n_features).
            
        Returns:
            Predicted class probabilities, a 2D array of shape (n_samples, n_classes).
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def save(self, path: str | Path) -> None:
        """
        Saves the scaler and model to the specified path.

        Args:
            path: The path where the scaler and model will be saved.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path / "scaler.pkl")
        joblib.dump(self.model, path / "model.pkl")

    @classmethod
    def load(cls, path: str | Path) -> "LogisticRegressionBaseline":
        """
        Loads the scaler and model from the specified path.

        Args:
            path: The path from which to load the scaler and model.

        Returns:
            An instance of LogisticRegressionBaseline with the loaded scaler and model.
        """
        path = Path(path)
        obj = cls.__new__(cls)
        obj.scaler = joblib.load(path / "scaler.pkl")
        obj.model = joblib.load(path / "model.pkl")
        return obj

