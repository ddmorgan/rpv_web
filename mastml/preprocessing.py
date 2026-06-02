from __future__ import annotations

from typing import Any


class SklearnPreprocessor:
    """Small shim for unpickling MAST-ML StandardScaler artifacts.

    The upstream pickles store a sklearn scaler under ``preprocessor``. The
    full mastml package is large, so this compatibility class provides the
    transform API needed for inference.
    """

    def __init__(self, preprocessor: Any | None = None, as_frame: bool = False):
        self.preprocessor = preprocessor
        self.as_frame = as_frame

    def fit(self, x: Any, y: Any | None = None) -> "SklearnPreprocessor":
        self.preprocessor.fit(x, y)
        return self

    def transform(self, x: Any) -> Any:
        return self.preprocessor.transform(x)

    def fit_transform(self, x: Any, y: Any | None = None) -> Any:
        return self.preprocessor.fit_transform(x, y)
