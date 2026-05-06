from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .config import NF2Config
from .modules import NF2Linear, NF2LoRALinear


def dtype_from_name(name: str) -> torch.dtype:
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[name]


def _get_parent(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def convert_model_to_nf2(
    model: nn.Module,
    config: NF2Config | None = None,
    skip_modules: tuple[str, ...] = ("lm_head",),
) -> list[str]:
    """Replace Linear layers in a Hugging Face model with NF2Linear layers in-place."""

    config = config or NF2Config()
    converted: list[str] = []
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(name == skip or name.startswith(skip + ".") for skip in skip_modules):
            continue
        targets.append((name, module))
    for name, module in targets:
        parent, attr = _get_parent(model, name)
        nf2 = NF2Linear.from_linear(module, config=config, compute_dtype=module.weight.dtype)
        nf2.to(device=module.weight.device)
        setattr(parent, attr, nf2)
        converted.append(name)
    return converted


def save_nf2_model(model: nn.Module, tokenizer, output_dir: str | Path, config: NF2Config, converted: list[str] | None = None) -> None:
    """Save NF2 model state plus enough metadata to reload the original HF architecture."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_modules = {}
    if converted is None:
        converted = []
        for name, module in model.named_modules():
            if isinstance(module, NF2LoRALinear):
                converted.append(name)
                lora_modules[name] = {"rank": module.rank, "alpha": module.alpha}
            elif isinstance(module, NF2Linear):
                converted.append(name)
    config.save(output_dir / "nf2_config.json")
    torch.save(model.state_dict(), output_dir / "nf2_model.pt")
    metadata = {"converted_modules": converted or [], "base_model_id": config.base_model_id, "lora_modules": lora_modules}
    (output_dir / "nf2_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    if hasattr(model, "config"):
        model.config.save_pretrained(output_dir)


def load_nf2_model(
    model_dir: str | Path,
    device: str | torch.device = "auto",
    dtype: str | torch.dtype | None = None,
    trust_remote_code: bool = False,
) -> tuple[nn.Module, object]:
    """Load an NF2 checkpoint by rebuilding the base architecture then applying NF2 modules."""

    model_dir = Path(model_dir)
    nf2_config = NF2Config.load(model_dir / "nf2_config.json")
    metadata_path = model_dir / "nf2_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    base_model_id = metadata.get("base_model_id") or nf2_config.base_model_id or str(model_dir)
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else dtype_from_name(dtype or nf2_config.transformers_dtype)
    hf_config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_config(hf_config, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
    state = torch.load(model_dir / "nf2_model.pt", map_location="cpu")
    converted = metadata.get("converted_modules") or []
    lora_modules = metadata.get("lora_modules") or {}
    for name in converted:
        parent, attr = _get_parent(model, name)
        old = getattr(parent, attr)
        nf2 = NF2Linear(old.in_features, old.out_features, old.bias is not None, nf2_config, torch_dtype)
        state_prefix = f"{name}.base." if name in lora_modules else f"{name}."
        if state_prefix + "qweight" in state:
            nf2.qweight = state[state_prefix + "qweight"].clone()
            nf2.scale = nn.Parameter(state[state_prefix + "scale"].clone(), requires_grad=False)
            nf2.offset = nn.Parameter(state[state_prefix + "offset"].clone(), requires_grad=False)
        if name in lora_modules:
            spec = lora_modules[name]
            nf2 = NF2LoRALinear(nf2, rank=int(spec["rank"]), alpha=float(spec["alpha"]))
        setattr(parent, attr, nf2)
    model.load_state_dict(state, strict=False)
    if device != "auto":
        model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    return model, tokenizer
