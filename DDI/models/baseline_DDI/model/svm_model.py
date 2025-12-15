import pickle
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score


class SVMModel:
    """
    One-vs-Rest Linear SVM model for multi-class DDI classification.
    """

    def __init__(self,
                 C=1.0,
                 max_iter=5000,
                 class_weight=None):
        """
        Args:
            C: Regularization strength
            max_iter: Max training iterations
            class_weight: None or "balanced"
        """
        self.model = OneVsRestClassifier(
            LinearSVC(
                C=C,
                max_iter=max_iter,
                class_weight="balanced"
            )
        )

    def train(self, X_train, y_train, X_valid, y_valid):
        """
        Train SVM model and evaluate on validation set.
        Returns validation accuracy and macro-F1.
        """
        print("[INFO] Training One-vs-Rest Linear SVM ...")
        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_valid)
        acc = accuracy_score(y_valid, preds)
        f1 = f1_score(y_valid, preds, average="macro")

        print(f"[INFO] Validation — Acc={acc:.4f}, Macro-F1={f1:.4f}")
        return acc, f1

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"[INFO] SVM model saved to {path}")

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            model = pickle.load(f)

        wrapper = SVMModel()
        wrapper.model = model
        print(f"[INFO] SVM model loaded from {path}")
        return wrapper
