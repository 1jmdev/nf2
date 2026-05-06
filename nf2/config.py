from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal


@dataclass(slots=True)
class NF2Config:
    """Configuration for scalar NF2 block quantization."""

    block_size: int = 64
    scale_dtype: str = "float16"
    offset_dtype: str = "float16"
    codebook: tuple[float, float, float, float] = (-1.0, -0.254917, 0.254917, 1.0)
    quantize_embeddings: bool = False
    target_module_types: tuple[str, ...] = ("Linear",)
    base_model_id: str | None = None
    transformers_dtype: str = "bfloat16"
    format_version: int = 1
    optional_hadamard: bool = False
    layout: Literal["flat-blocks"] = "flat-blocks"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["codebook"] = list(self.codebook)
        data["target_module_types"] = list(self.target_module_types)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "NF2Config":
        fixed = dict(data)
        if "codebook" in fixed:
            fixed["codebook"] = tuple(float(x) for x in fixed["codebook"])
        if "target_module_types" in fixed:
            fixed["target_module_types"] = tuple(fixed["target_module_types"])
        return cls(**fixed)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NF2Config":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
