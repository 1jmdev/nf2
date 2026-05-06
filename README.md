# NF2

NF2 is a Python-only reference project for converting Hugging Face causal language models from FP16/BF16 weights into scalar NormalFloat 2-bit weights, then recovering quality with LoRA adapters trained by KL distillation against the original teacher model.

Pipeline:

```text
Input weights (FP16/BF16)
  -> block normalize with scale + offset
  -> NF2 encode with 4-level codebook
  -> recovery FT with LoRA + KL loss
  -> NF2+ model: 2 bpw packed weights + adapters
```

The implementation is correctness-first and uses normal PyTorch modules. It does not require custom CUDA kernels, but generation will be slower than a fused production kernel because weights are dequantized inside each `NF2Linear` forward pass.

## Format

NF2 stores each weight as a 2-bit index into this normalized codebook:

```python
[-1.0, -0.254917, 0.254917, 1.0]
```

These are the quartile conditional means `[-1.271, -0.324, 0.324, 1.271]` normalized by their absolute maximum. With absmax block scaling, this normalized codebook is required to avoid reconstructing values outside the block range.

Weights are grouped into blocks of 64. Each block stores an FP16 scale and FP16 offset.

Reconstruction:

```text
w_i ~= codebook[q_i] * scale_b + offset_b
```

Raw packed weight cost is 2 bits/weight. Scale and offset add 32 bits per 64-weight block, so the reference format is about 2.5 bits/weight before LoRA adapters and unquantized skipped layers such as `lm_head`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

You need Hugging Face access for `meta-llama/Llama-3.2-1B`:

```bash
huggingface-cli login
```

## Convert Llama 3.2 1B

```bash
python examples/llama32-1b.py convert --output-dir outputs/llama32-1b-nf2
```

Equivalent CLI:

```bash
nf2 convert --model-id meta-llama/Llama-3.2-1B --output-dir outputs/llama32-1b-nf2
```

By default the converter quantizes all `torch.nn.Linear` layers except `lm_head`. It stores:

```text
outputs/llama32-1b-nf2/nf2_config.json
outputs/llama32-1b-nf2/nf2_metadata.json
outputs/llama32-1b-nf2/nf2_model.pt
outputs/llama32-1b-nf2/tokenizer files
```

## Recovery Fine-Tuning

The recovery step adds LoRA adapters to every NF2 linear layer, freezes NF2 weights, and minimizes KL divergence between the NF2+ student logits and the original BF16/FP16 teacher logits.

Default dataset is about 50 MB streamed from `HuggingFaceFW/fineweb-edu`, config `sample-10BT`.

```bash
python examples/llama32-1b.py recover \
  --model-dir outputs/llama32-1b-nf2 \
  --output-dir outputs/llama32-1b-nf2-plus \
  --target-bytes 50000000 \
  --max-steps 1000 \
  --lora-rank 8
```

For a quick smoke run, use fewer bytes and steps:

```bash
python examples/llama32-1b.py recover --target-bytes 1000000 --max-steps 5
```

## Generate

```bash
python examples/llama32-1b.py generate \
  --model-dir outputs/llama32-1b-nf2-plus \
  --prompt "Write a short explanation of NF2 quantization."
```

Equivalent CLI:

```bash
nf2 generate --model-dir outputs/llama32-1b-nf2-plus --prompt "Explain NF2."
```

## Python API

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nf2 import NF2Config, convert_model_to_nf2, save_nf2_model

model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

config = NF2Config(base_model_id=model_id)
converted = convert_model_to_nf2(model, config=config, skip_modules=("lm_head",))
save_nf2_model(model, tokenizer, "outputs/llama32-1b-nf2", config, converted)
```

## Project Layout

```text
nf2/config.py       format configuration
nf2/quant.py        NF2 codebook, quantize, pack, unpack, dequantize
nf2/modules.py      NF2Linear and NF2LoRALinear
nf2/convert.py      Hugging Face conversion and checkpoint load/save
nf2/recovery.py     LoRA + KL distillation recovery
nf2/data.py         streamed calibration text loading
nf2/generation.py   generation helper
nf2/cli.py          command line interface
examples/llama32-1b.py
```

## Notes

This is a complete research/reference implementation, not a production inference engine. For maximum throughput, the next step is a fused NF2 matmul kernel that performs table lookup, scale/offset reconstruction, and matrix multiplication without materializing dense weights.
