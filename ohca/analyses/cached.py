from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any, Callable


class Cached:

    # Analyses persist their named fields as one pickled dict; save() never overwrites.

    def save(self, path: Path) -> None:
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(vars(self), f)

    @classmethod
    def load(cls, path: Path) -> Any:
        with open(path, "rb") as f:
            return cls(**pickle.load(f))

    @classmethod
    def cached(cls, path: Path, create: Callable[[], Any]) -> Any:
        obj = cls.load(path) if path.exists() else create()
        obj.save(path)
        return obj
