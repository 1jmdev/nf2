from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import NF2Config
from .quant import NF2_CODEBOOK, dequantize_nf2, quantize_nf2


class NF2Linear(nn.Module):
    """Drop-in Linear layer backed by packed scalar NF2 weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: NF2Config | None = None,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = (config or NF2Config()).block_size
        self.compute_dtype = compute_dtype
        self.register_buffer("qweight", torch.empty(0, dtype=torch.uint8), persistent=True)
        self.scale = nn.Parameter(torch.empty(0, dtype=torch.float16), requires_grad=False)
        self.offset = nn.Parameter(torch.empty(0, dtype=torch.float16), requires_grad=False)
        self.register_buffer("codebook", torch.tensor((config or NF2Config()).codebook, dtype=torch.float32), persistent=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=compute_dtype), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        config: NF2Config | None = None,
        compute_dtype: torch.dtype | None = None,
    ) -> "NF2Linear":
        compute_dtype = compute_dtype or linear.weight.dtype
        layer = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            config=config,
            compute_dtype=compute_dtype,
        )
        q = quantize_nf2(
            linear.weight.data,
            block_size=layer.block_size,
            codebook=layer.codebook,
            quant_iters=(config or NF2Config()).quant_iters,
        )
        layer.qweight = q["qweight"]
        layer.scale = nn.Parameter(q["scale"], requires_grad=False)
        layer.offset = nn.Parameter(q["offset"], requires_grad=False)
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data.to(compute_dtype))
        return layer

    def dequantize_weight(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        return dequantize_nf2(
            self.qweight,
            self.scale,
            self.offset,
            (self.out_features, self.in_features),
            block_size=self.block_size,
            codebook=self.codebook,
            dtype=dtype or self.compute_dtype,
            device=self.qweight.device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize_weight(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


class NF2LoRALinear(nn.Module):
    """Trainable LoRA recovery adapter on top of a frozen NF2Linear."""

    def __init__(self, base: NF2Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        device = base.qweight.device
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=base.compute_dtype, device=device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=base.compute_dtype, device=device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for param in self.base.parameters():
            param.requires_grad_(False)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        a = self.lora_a.to(dtype=x.dtype)
        b = self.lora_b.to(dtype=x.dtype)
        return out + F.linear(F.linear(self.dropout(x), a), b) * self.scaling


def _get_parent(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def iter_named_nf2_linears(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, NF2Linear):
            yield name, module


def add_lora_adapters(model: nn.Module, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> list[str]:
    """Wrap every NF2Linear in an NF2LoRALinear and return wrapped module names."""

    names = [name for name, module in model.named_modules() if isinstance(module, NF2Linear)]
    for name in names:
        parent, attr = _get_parent(model, name)
        setattr(parent, attr, NF2LoRALinear(getattr(parent, attr), rank=rank, alpha=alpha, dropout=dropout))
    return names


def merge_lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only LoRA tensors for compact adapter checkpoints."""

    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if "lora_" in name}


def iter_top_level_nf2_modules(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, NF2LoRALinear):
            yield name, module
        elif isinstance(module, NF2Linear) and not name.endswith(".base"):
            yield name, module


def freeze_non_lora(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad_("lora_" in name)
