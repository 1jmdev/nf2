from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from .convert import dtype_from_name
from .data import load_text_samples, token_batches
from .modules import NF2Linear, add_lora_adapters, freeze_non_lora, merge_lora_state


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
    scale_offset_learning_rate: float = 1e-6
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    temperature: float = 1.0
    top_k: int = 256
    ce_weight: float = 0.1
    refine_scale_offset: bool = False
    dtype: str = "bfloat16"
    grad_clip: float = 1.0
    save_adapter_path: str | None = None


def _kl_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    top_k: int = 256,
    ce_weight: float = 0.1,
) -> torch.Tensor:
    student = student_logits[:, :-1, :].float()
    teacher = teacher_logits[:, :-1, :].float()
    vocab = teacher.shape[-1]
    if top_k and 0 < top_k < vocab:
        teacher_top = torch.topk(teacher, k=top_k, dim=-1)
        teacher_probs = F.softmax(teacher / temperature, dim=-1)
        student_probs = F.softmax(student / temperature, dim=-1)
        target_top = torch.gather(teacher_probs, dim=-1, index=teacher_top.indices)
        student_top = torch.gather(student_probs, dim=-1, index=teacher_top.indices)
        target_other = (1.0 - target_top.sum(dim=-1, keepdim=True)).clamp_min(1e-8)
        student_other = (1.0 - student_top.sum(dim=-1, keepdim=True)).clamp_min(1e-8)
        target = torch.cat([target_top, target_other], dim=-1).clamp_min(1e-8)
        student_dist = torch.cat([student_top, student_other], dim=-1).clamp_min(1e-8)
        kl = F.kl_div(student_dist.log(), target, reduction="batchmean")
        kl = kl * (temperature**2) / max(1, student.shape[1])
    else:
        kl = F.kl_div(
            F.log_softmax(student / temperature, dim=-1),
            F.softmax(teacher / temperature, dim=-1),
            reduction="batchmean",
        )
        kl = kl * (temperature**2) / max(1, student.shape[1])
    if ce_weight <= 0:
        return kl
    hard_targets = teacher.argmax(dim=-1)
    ce = F.cross_entropy(student.reshape(-1, student.shape[-1]), hard_targets.reshape(-1))
    return kl + ce * ce_weight


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
        for module in student_model.modules():
            if isinstance(module, NF2Linear):
                module.scale = torch.nn.Parameter(module.scale.detach().float(), requires_grad=True)
                module.offset = torch.nn.Parameter(module.offset.detach().float(), requires_grad=True)

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_id,
        dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    ).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    lora_params = []
    scale_offset_params = []
    for name, param in student_model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("scale") or name.endswith("offset"):
            scale_offset_params.append(param)
        else:
            lora_params.append(param)
    param_groups = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": config.learning_rate})
    if scale_offset_params:
        param_groups.append({"params": scale_offset_params, "lr": config.scale_offset_learning_rate, "weight_decay": 0.0})
    trainable = lora_params + scale_offset_params
    optimizer = torch.optim.AdamW(param_groups)
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
        loss = _kl_distill_loss(student_logits, teacher_logits, config.temperature, config.top_k, config.ce_weight)
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
