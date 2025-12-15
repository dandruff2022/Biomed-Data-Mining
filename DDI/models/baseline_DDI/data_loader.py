import json
from typing import List, Tuple


class JSONDataLoader:
    """
    Load dataset for DDI classification.
    Each JSON file is a list of objects:
    {
        "id": int,
        "sentence": str,
        "answer": str,
        "label": int
    }
    """

    def __init__(self, train_path: str, valid_path: str, test_path: str):
        self.paths = {
            "train": train_path,
            "valid": valid_path,
            "test": test_path
        }

    def _load_json(self, path: str):
        """Load a JSON file containing a list of dicts."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def load_split(self, split: str) -> Tuple[List[str], List[int]]:
        """
        Return:
            texts:  List[str] — the "sentence" field
            labels: List[int] — the "label" field
        """
        if split not in self.paths:
            raise ValueError(f"Invalid split name: {split}. Must be train/valid/test.")

        data = self._load_json(self.paths[split])

        texts = []
        labels = []

        for item in data:
            texts.append(item["sentence"])
            labels.append(int(item["label"]))

        return texts, labels
