from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True)
class PreprocessSpec:
    size: ClassVar[int] = 1024
    is_sem: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.is_sem, bool):
            raise ValueError("is_sem must be a bool")

    def to_meta(self) -> dict[str, Any]:
        return {"size": int(self.size), "is_sem": self.is_sem}

    @classmethod
    def from_meta(cls, meta: dict[str, Any] | None) -> "PreprocessSpec":
        if not meta or "preprocess" not in meta:
            return cls()
        value = meta["preprocess"]
        if not isinstance(value, dict):
            raise RuntimeError("Checkpoint preprocess metadata must be a mapping")
        try:
            if type(value["size"]) is not int or value["size"] != cls.size:
                raise ValueError("Input size is fixed at 1024")
            is_sem = (value["is_sem"] if "is_sem" in value else
                      {"rgb": False, "sem_features": True}[value["mode"]])
            return cls(
                is_sem=is_sem,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Invalid checkpoint preprocess metadata") from exc
