from dataclasses import dataclass
from typing import Any

from .convert import InputMode

MODES = {"rgb", "gray_repeat", "gray_features"}


@dataclass(frozen=True)
class PreprocessSpec:
    size: int = 1024
    mode: InputMode = "rgb"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.size, int)
            or isinstance(self.size, bool)
            or self.size <= 0
        ):
            raise ValueError("preprocess size must be a positive integer")
        if not isinstance(self.mode, str) or self.mode not in MODES:
            raise ValueError(f"Unsupported input mode: {self.mode!r}")

    def to_meta(self) -> dict[str, Any]:
        return {"size": int(self.size), "mode": self.mode}

    @classmethod
    def from_meta(cls, meta: dict[str, Any] | None) -> "PreprocessSpec":
        if not meta or "preprocess" not in meta:
            return cls()
        value = meta["preprocess"]
        if not isinstance(value, dict):
            raise RuntimeError("Checkpoint preprocess metadata must be a mapping")
        try:
            return cls(
                size=value["size"],
                mode=value["mode"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Invalid checkpoint preprocess metadata") from exc
