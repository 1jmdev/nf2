from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import NF2Config
from .convert import convert_model_to_nf2, dtype_from_name, load_nf2_model, save_nf2_model
from .generation import generate_text
from .recovery import RecoveryConfig, run_recovery_ft


def _convert(args: argparse.Namespace) -> None:
    dtype = dtype_from_name(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    config = NF2Config(
        block_size=args.block_size,
        quant_iters=args.quant_iters,
        base_model_id=args.model_id,
        transformers_dtype=args.dtype,
    )
    converted = convert_model_to_nf2(model, config=config, skip_modules=tuple(args.skip_module))
    save_nf2_model(model, tokenizer, args.output_dir, config, converted)
    print(f"Converted {len(converted)} Linear layers to NF2 and saved to {args.output_dir}")


def _generate(args: argparse.Namespace) -> None:
    model, tokenizer = load_nf2_model(args.model_dir, device=args.device, dtype=args.dtype, trust_remote_code=args.trust_remote_code)
    print(generate_text(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_p, args.device))


def _recover(args: argparse.Namespace) -> None:
    model, tokenizer = load_nf2_model(args.model_dir, device=args.device, dtype=args.dtype, trust_remote_code=args.trust_remote_code)
    cfg = RecoveryConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        target_bytes=args.target_bytes,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        scale_offset_learning_rate=args.scale_offset_learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        top_k=args.top_k,
        ce_weight=args.ce_weight,
        refine_scale_offset=args.refine_scale_offset,
        save_adapter_path=args.save_adapter_path,
        dtype=args.dtype,
    )
    metrics = run_recovery_ft(model, tokenizer, args.teacher_model_id, cfg, args.device, args.trust_remote_code)
    print(metrics)
    if args.output_dir:
        nf2_config = NF2Config.load(f"{args.model_dir}/nf2_config.json")
        save_nf2_model(model, tokenizer, args.output_dir, nf2_config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NF2 quantization and recovery tools")
    sub = parser.add_subparsers(required=True)

    c = sub.add_parser("convert", help="Convert an HF causal LM to NF2")
    c.add_argument("--model-id", default="meta-llama/Llama-3.2-1B")
    c.add_argument("--output-dir", required=True)
    c.add_argument("--dtype", default="bfloat16")
    c.add_argument("--device-map", default="auto")
    c.add_argument("--block-size", type=int, default=16)
    c.add_argument("--quant-iters", type=int, default=5)
    c.add_argument("--skip-module", action="append", default=["lm_head"])
    c.add_argument("--trust-remote-code", action="store_true")
    c.set_defaults(func=_convert)

    g = sub.add_parser("generate", help="Generate from an NF2 checkpoint")
    g.add_argument("--model-dir", required=True)
    g.add_argument("--prompt", required=True)
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--dtype", default="bfloat16")
    g.add_argument("--max-new-tokens", type=int, default=128)
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--trust-remote-code", action="store_true")
    g.set_defaults(func=_generate)

    r = sub.add_parser("recover", help="Run LoRA+KL recovery FT")
    r.add_argument("--model-dir", required=True)
    r.add_argument("--teacher-model-id", default="meta-llama/Llama-3.2-1B")
    r.add_argument("--output-dir")
    r.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    r.add_argument("--dtype", default="bfloat16")
    r.add_argument("--dataset-name", default="HuggingFaceFW/fineweb-edu")
    r.add_argument("--dataset-config", default="sample-10BT")
    r.add_argument("--target-bytes", type=int, default=50_000_000)
    r.add_argument("--sequence-length", type=int, default=1024)
    r.add_argument("--batch-size", type=int, default=1)
    r.add_argument("--max-steps", type=int, default=1000)
    r.add_argument("--learning-rate", type=float, default=2e-4)
    r.add_argument("--scale-offset-learning-rate", type=float, default=1e-6)
    r.add_argument("--lora-rank", type=int, default=8)
    r.add_argument("--lora-alpha", type=float, default=16.0)
    r.add_argument("--top-k", type=int, default=256)
    r.add_argument("--ce-weight", type=float, default=0.1)
    r.add_argument("--refine-scale-offset", action="store_true")
    r.add_argument("--save-adapter-path")
    r.add_argument("--trust-remote-code", action="store_true")
    r.set_defaults(func=_recover)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
