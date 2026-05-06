from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nf2 import NF2Config, RecoveryConfig, convert_model_to_nf2, generate_text, load_nf2_model, run_recovery_ft, save_nf2_model


MODEL_ID = "meta-llama/Llama-3.2-1B"


def convert(args: argparse.Namespace) -> None:
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype, device_map=args.device_map)
    config = NF2Config(block_size=args.block_size, base_model_id=args.model_id, transformers_dtype=args.dtype)
    converted = convert_model_to_nf2(model, config=config, skip_modules=("lm_head",))
    save_nf2_model(model, tokenizer, args.output_dir, config, converted)
    print(f"Saved NF2 checkpoint with {len(converted)} converted layers: {args.output_dir}")


def recover(args: argparse.Namespace) -> None:
    model, tokenizer = load_nf2_model(args.model_dir, device=args.device, dtype=args.dtype)
    cfg = RecoveryConfig(
        max_steps=args.max_steps,
        target_bytes=args.target_bytes,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        save_adapter_path=args.adapter_path,
        dtype=args.dtype,
    )
    metrics = run_recovery_ft(model, tokenizer, args.model_id, cfg, device=args.device)
    nf2_config = NF2Config.load(f"{args.model_dir}/nf2_config.json")
    save_nf2_model(model, tokenizer, args.output_dir, nf2_config)
    print(f"Recovery metrics: {metrics}")
    print(f"Saved NF2+LoRA checkpoint: {args.output_dir}")


def generate(args: argparse.Namespace) -> None:
    model, tokenizer = load_nf2_model(args.model_dir, device=args.device, dtype=args.dtype)
    print(generate_text(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_p, args.device))


def main() -> None:
    parser = argparse.ArgumentParser(description="Llama 3.2 1B NF2 pipeline example")
    sub = parser.add_subparsers(required=True)

    c = sub.add_parser("convert")
    c.add_argument("--model-id", default=MODEL_ID)
    c.add_argument("--output-dir", default="outputs/llama32-1b-nf2")
    c.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    c.add_argument("--device-map", default="auto")
    c.add_argument("--block-size", type=int, default=64)
    c.set_defaults(func=convert)

    r = sub.add_parser("recover")
    r.add_argument("--model-id", default=MODEL_ID)
    r.add_argument("--model-dir", default="outputs/llama32-1b-nf2")
    r.add_argument("--output-dir", default="outputs/llama32-1b-nf2-plus")
    r.add_argument("--adapter-path", default="outputs/llama32-1b-nf2-plus/lora_adapters.pt")
    r.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    r.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    r.add_argument("--target-bytes", type=int, default=50_000_000)
    r.add_argument("--sequence-length", type=int, default=1024)
    r.add_argument("--batch-size", type=int, default=1)
    r.add_argument("--max-steps", type=int, default=1000)
    r.add_argument("--learning-rate", type=float, default=2e-4)
    r.add_argument("--lora-rank", type=int, default=8)
    r.add_argument("--lora-alpha", type=float, default=16.0)
    r.set_defaults(func=recover)

    g = sub.add_parser("generate")
    g.add_argument("--model-dir", default="outputs/llama32-1b-nf2-plus")
    g.add_argument("--prompt", default="Explain NF2 quantization in one paragraph.")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    g.add_argument("--max-new-tokens", type=int, default=128)
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--top-p", type=float, default=0.95)
    g.set_defaults(func=generate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
