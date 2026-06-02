from __future__ import annotations

from typing import Any


class SklearnModel:
    """Small shim for unpickling MAST-ML sklearn model artifacts."""

    def __init__(self, model: Any | None = None):
        self.model = model

    def fit(self, x: Any, y: Any, **kwargs: Any) -> "SklearnModel":
        self.model.fit(x, y, **kwargs)
        return self

    def predict(self, x: Any, as_frame: bool = False) -> Any:
        return self.model.predict(x)
