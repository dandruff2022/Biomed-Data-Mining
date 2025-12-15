import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFVectorizer:
    """
    Wrapper for sklearn TfidfVectorizer with save/load support.
    """

    def __init__(self,
                 max_features: int = 50000,
                 ngram_range=(1, 2),
                 lowercase: bool = True,
                 stop_words="english"):
        """
        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: Use unigrams/bigrams (default)
            lowercase: Convert text to lowercase
            stop_words: Use English stopwords if needed ("english")
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=lowercase,
            stop_words=stop_words
        )

    def fit_transform(self, texts):
        """
        Fit the TF-IDF model on training texts and return the matrix.
        """
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        """
        Transform new texts using the fitted vectorizer.
        """
        return self.vectorizer.transform(texts)

   
    def save(self, path: str):
        """
        Save vectorizer into a pickle file.
        """
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        print(f"[INFO] TF-IDF vectorizer saved to {path}")

    @staticmethod
    def load(path: str):
        """
        Load a saved TF-IDF vectorizer.
        """
        with open(path, "rb") as f:
            vec = pickle.load(f)

        wrapper = TFIDFVectorizer()
        wrapper.vectorizer = vec
        print(f"[INFO] TF-IDF vectorizer loaded from {path}")
        return wrapper
