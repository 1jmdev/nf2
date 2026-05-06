from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from .convert import dtype_from_name
from .data import load_text_samples, token_batches
from .modules import add_lora_adapters, freeze_non_lora, merge_lora_state


@dataclass(slots=True)
class RecoveryConfig:
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str | None = "sample-10BT"
    split: str = "train"
    text_column: str = "text"
    target_bytes: int = 50_000_000
    sequence_length: int = 1024
    batch_size: int = 1
    max_steps: int = 1000
    learning_rate: float = 2e-4
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    temperature: float = 1.0
    refine_scale_offset: bool = False
    dtype: str = "bfloat16"
    grad_clip: float = 1.0
    save_adapter_path: str | None = None


def _kl_distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    s = student_logits[:, :-1, :].float() / temperature
    t = teacher_logits[:, :-1, :].float() / temperature
    loss = F.kl_div(F.log_softmax(s, dim=-1), F.softmax(t, dim=-1), reduction="batchmean")
    return loss * (temperature**2) / max(1, s.shape[1])


def run_recovery_ft(
    student_model,
    tokenizer,
    teacher_model_id: str,
    config: RecoveryConfig | None = None,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    trust_remote_code: bool = False,
) -> dict[str, float]:
    """Run LoRA+KL recovery fine-tuning against an FP/BF16 teacher model."""

    config = config or RecoveryConfig()
    dtype = dtype_from_name(config.dtype)
    student_model.to(device)
    add_lora_adapters(student_model, rank=config.lora_rank, alpha=config.lora_alpha, dropout=config.lora_dropout)
    freeze_non_lora(student_model)
    if config.refine_scale_offset:
        for name, param in student_model.named_parameters():
            if name.endswith("scale") or name.endswith("offset"):
                param.requires_grad_(True)

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_id,
        dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    ).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    trainable = [p for p in student_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    texts = load_text_samples(
        config.dataset_name,
        config.dataset_config,
        config.split,
        config.text_column,
        config.target_bytes,
        streaming=True,
    )

    student_model.train()
    losses: list[float] = []
    batches = token_batches(
        tokenizer,
        texts,
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
        max_batches=config.max_steps,
        device=device,
    )
    for batch in tqdm(batches, total=config.max_steps, desc="NF2 recovery FT"):
        with torch.no_grad():
            teacher_logits = teacher(**batch, use_cache=False).logits
        student_logits = student_model(**batch, use_cache=False).logits
        loss = _kl_distill_loss(student_logits, teacher_logits, config.temperature)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    if config.save_adapter_path:
        path = Path(config.save_adapter_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(merge_lora_state(student_model), path)
    return {"loss": sum(losses) / max(1, len(losses)), "steps": float(len(losses))}
