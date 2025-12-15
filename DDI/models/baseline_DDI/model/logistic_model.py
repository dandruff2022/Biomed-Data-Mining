import pickle
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score


class LogisticModel:
    """
    Logistic Regression (OvR) + sentence length feature
    """

    def __init__(self,
                 C=1.0,
                 max_iter=2000,
                 n_jobs=-1):

        self.model = OneVsRestClassifier(
            LogisticRegression(
                C=C,
                max_iter=max_iter,
                n_jobs=n_jobs,
                solver="liblinear",
                class_weight="balanced"
            )
        )

        # will store mean/std for length normalization
        self.len_mean = None
        self.len_std = None

    def _compute_length_feature(self, texts):
        lengths = np.array([len(t.split()) for t in texts]).reshape(-1, 1)
        return lengths

    def _normalize_length(self, lengths):
        return (lengths - self.len_mean) / (self.len_std + 1e-8)

    def _combine_features(self, X, lengths):
        lengths_sparse = csr_matrix(lengths)
        return hstack([X, lengths_sparse])

    def train(self, train_texts, X_train, y_train,
              valid_texts, X_valid, y_valid):

        # compute sentence lengths (train)
        train_lengths = self._compute_length_feature(train_texts)

        # compute normalization stats
        self.len_mean = train_lengths.mean()
        self.len_std = train_lengths.std()

        # normalize lengths
        train_lengths_norm = self._normalize_length(train_lengths)

        # combine features
        X_train_aug = self._combine_features(X_train, train_lengths_norm)

        # validation lengths
        valid_lengths = self._compute_length_feature(valid_texts)
        valid_lengths_norm = self._normalize_length(valid_lengths)
        X_valid_aug = self._combine_features(X_valid, valid_lengths_norm)

        print("[INFO] Fitting Logistic Regression (with length feature)...")
        self.model.fit(X_train_aug, y_train)

        preds = self.model.predict(X_valid_aug)
        acc = accuracy_score(y_valid, preds)
        f1 = f1_score(y_valid, preds, average="macro")

        print(f"[INFO] Validation — Acc={acc:.4f}, Macro-F1={f1:.4f}")
        return acc, f1

    # Prediction
    def predict(self, texts, X):
        lengths = self._compute_length_feature(texts)
        lengths_norm = self._normalize_length(lengths)
        X_aug = self._combine_features(X, lengths_norm)
        return self.model.predict(X_aug)

    # Save 
    def save(self, path):
        obj = {
            "model": self.model,
            "len_mean": self.len_mean,
            "len_std": self.len_std,
        }
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"[INFO] Model saved to {path}")

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)

        wrapper = LogisticModel()
        wrapper.model = obj["model"]
        wrapper.len_mean = obj["len_mean"]
        wrapper.len_std = obj["len_std"]

        print(f"[INFO] Model loaded from {path}")
        return wrapper
