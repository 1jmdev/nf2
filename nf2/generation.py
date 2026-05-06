from __future__ import annotations

import torch


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    device: str | torch.device | None = None,
) -> str:
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        kwargs.update({"temperature": temperature, "top_p": top_p})
    out = model.generate(**inputs, **kwargs)
    return tokenizer.decode(out[0], skip_special_tokens=True)
