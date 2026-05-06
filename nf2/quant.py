from __future__ import annotations

import math

import torch

# Quartile conditional means of N(0, 1), normalized by absmax so block values
# reconstructed after absmax scaling cannot overshoot the original block range.
NF2_CODEBOOK = torch.tensor([-1.0, -0.254917, 0.254917, 1.0], dtype=torch.float32)


def pack_2bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack uint8 NF2 indices into bytes, four 2-bit values per byte."""

    flat = indices.to(torch.uint8).flatten()
    pad = (-flat.numel()) % 4
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    flat = flat.view(-1, 4)
    return flat[:, 0] | (flat[:, 1] << 2) | (flat[:, 2] << 4) | (flat[:, 3] << 6)


def unpack_2bit(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack bytes produced by pack_2bit into uint8 indices."""

    packed = packed.to(torch.uint8).flatten()
    out = torch.empty(packed.numel() * 4, dtype=torch.uint8, device=packed.device)
    out[0::4] = packed & 0b00000011
    out[1::4] = (packed >> 2) & 0b00000011
    out[2::4] = (packed >> 4) & 0b00000011
    out[3::4] = (packed >> 6) & 0b00000011
    return out[:count]


def _pad_blocks(x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int]:
    pad = (-x.numel()) % block_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    return x.view(-1, block_size), pad


@torch.no_grad()
def quantize_nf2(
    weight: torch.Tensor,
    block_size: int = 64,
    codebook: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | tuple[int, ...] | int]:
    """Quantize a weight tensor to NF2 indices plus per-block scale and offset."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    original_shape = tuple(weight.shape)
    codebook = (NF2_CODEBOOK if codebook is None else codebook).to(weight.device, torch.float32)
    flat = weight.detach().to(torch.float32).flatten()
    blocks, pad = _pad_blocks(flat, block_size)
    offsets = blocks.mean(dim=1)
    centered = blocks - offsets[:, None]
    scales = centered.abs().amax(dim=1).clamp_min(1e-8)
    normalized = centered / scales[:, None]
    distances = (normalized[..., None] - codebook.view(1, 1, 4)).abs()
    indices = distances.argmin(dim=-1).to(torch.uint8).flatten()
    if pad:
        indices = indices[:-pad]
    return {
        "qweight": pack_2bit(indices).cpu(),
        "scale": scales.to(torch.float16).cpu(),
        "offset": offsets.to(torch.float16).cpu(),
        "shape": original_shape,
        "numel": flat.numel(),
        "block_size": block_size,
    }


def dequantize_nf2(
    qweight: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    shape: tuple[int, ...],
    block_size: int = 64,
    codebook: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Dequantize NF2 packed weights back to a dense tensor."""

    count = math.prod(shape)
    device = device or qweight.device
    dtype = dtype or torch.float16
    codebook = (NF2_CODEBOOK if codebook is None else codebook).to(device=device, dtype=torch.float32)
    indices = unpack_2bit(qweight.to(device), count).to(torch.long)
    values = codebook[indices]
    pad = (-count) % block_size
    if pad:
        values = torch.nn.functional.pad(values, (0, pad))
    blocks = values.view(-1, block_size)
    dense = blocks * scale.to(device=device, dtype=torch.float32)[:, None]
    dense = dense + offset.to(device=device, dtype=torch.float32)[:, None]
    dense = dense.flatten()[:count].view(shape)
    return dense.to(dtype)
