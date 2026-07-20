"""
Lightweight complexity-based router: decides whether a task should go to the
fast student model or fall back to the full teacher pipeline.
"""
from sklearn.linear_model import LogisticRegression
import re
import pickle


def extract_features(problem: str) -> list[float]:
    return [
        len(problem),
        problem.count("\n"),
        len(re.findall(r"\bfor\b|\bwhile\b", problem)),
        len(re.findall(r"def \w+\(", problem)),
    ]


class ComplexityRouter:
    def __init__(self):
        self.model = LogisticRegression()
        self.fitted = False

    def fit(self, problems: list[str], labels: list[int]):
        X = [extract_features(p) for p in problems]
        self.model.fit(X, labels)
        self.fitted = True

    def predict(self, problem: str) -> str:
        if not self.fitted:
            return "student"
        X = [extract_features(problem)]
        pred = self.model.predict(X)[0]
        return "teacher" if pred == 1 else "student"

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "ComplexityRouter":
        with open(path, "rb") as f:
            return pickle.load(f)