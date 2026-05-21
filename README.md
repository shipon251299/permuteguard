# PermuteGuard Research Code

Research code and experiment assets for **PermuteGuard**, a parameter-permutation watermarking framework for open-source large language models.

## Overview

This repository contains the code used to run the main experiment pipeline described in the PermuteGuard manuscript:

- **Stage 1**: watermark embedding fidelity and baseline robustness
- **Stage 2**: comparative robustness against a sign-bit baseline on stable benchmark subsets
- **Stage 3**: quality-gated utility validation under practically relevant attacks

The repository also includes benchmark subsets, prompt files, plotting utilities, intermediate experiment metadata, and saved result folders.

## Repository Structure

```text
permuteguard_research_final/
├── benchmarks/                         # Local benchmark JSONL files
│   ├── arc_easy.jsonl
│   ├── boolq.jsonl
│   ├── hellaswag.jsonl
│   └── piqa.jsonl
├── src/
│   ├── open_source_model_loader.py     # Local model loading utilities
│   └── permuteguard_large.py           # Core PermuteGuard implementation
├── stage2_benchmark_eval.py            # Benchmark normalization/evaluation helpers
├── stage2_weight_baselines.py          # Sign-bit baseline implementation
├── run_open_source_model.py            # Stage 1 experiment runner
├── run_stage2_comparison.py            # Stage 2 comparison runner
├── run_stage3_utility_validation.py    # Stage 3 utility validation (older runner)
├── run_stage3_utility_validation_v2.py # Stage 3 utility validation (main updated runner)
├── prepare_stage2_benchmarks.py        # Benchmark preparation utility
├── plot_experiment_results.py          # Result plotting utility
├── stage3_utility_prompts*.jsonl       # Prompt sets used in Stage 3
├── results/                            # Stage 1 result folders
├── stage2_results/                     # Stage 2 comparison and sensitivity outputs
├── stage3_results/                     # Stage 3 utility outputs
├── experiments_v2/                     # Saved experiment metadata and fingerprints
└── logs/                               # Run logs
```

## Environment

The uploaded archive indicates a Python 3.10 workflow (for example, cached bytecode files were generated with CPython 3.10). A practical starting point is:

- Python **3.10**
- PyTorch
- Transformers
- Datasets
- NumPy
- Pandas
- Matplotlib
- BitsAndBytes support when using quantized or memory-constrained model loading

A typical install flow is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers datasets numpy pandas matplotlib bitsandbytes
```

> Note: exact package versions were not bundled in the uploaded archive. Before public release, pin and test the exact dependency versions used for the final manuscript experiments.

## Supported Model Types

The loader currently supports:

- `gpt2`
- `tinyllama`
- `mistral`
- `falcon`
- `bloom`
- `opt`

Models are expected to be available **locally**. Large model checkpoints and virtual environments were intentionally excluded from the shared archive.

## Main Scripts

### 1. Stage 1: Watermark embedding and robustness

```bash
python run_open_source_model.py \
  --model /path/to/local/model \
  --model-type mistral \
  --output results/mistral7b_v4 \
  --trials 3 \
  --load-policy auto \
  --watermark-key YOUR_SECRET_KEY \
  --permutation-strength 0.002 \
  --verification-threshold 0.80 \
  --max-layers 8 \
  --min-layer-size 50000
```

Key Stage 1 options:

- `--model`: local checkpoint directory
- `--model-type`: model family
- `--output`: output directory
- `--trials`: number of trials
- `--load-policy`: one of `auto`, `fp32_single`, `fp16_auto`, `int8_auto`
- `--watermark-key`: secret watermark key
- `--permutation-strength`: watermark intensity
- `--verification-threshold`: verification threshold
- `--max-layers`: maximum number of selected layers
- `--min-layer-size`: minimum eligible layer size

### 2. Stage 2: PermuteGuard vs sign-bit comparison

```bash
python run_stage2_comparison.py \
  --model /path/to/local/model \
  --model-type mistral \
  --task-files benchmarks/piqa.jsonl,benchmarks/boolq.jsonl,benchmarks/arc_easy.jsonl \
  --output stage2_results/mistral_cmp_lora_v2 \
  --methods permutation,signbit \
  --attacks none,pruning_10,quantization_8bit,shuffle_05,lora_medium \
  --trials 3 \
  --max-examples 50 \
  --watermark-key YOUR_SECRET_KEY \
  --permutation-strength 0.002 \
  --verification-threshold 0.80 \
  --max-layers 8
```

### 3. Stage 3: Utility validation

```bash
python run_stage3_utility_validation_v2.py \
  --model /path/to/local/model \
  --model-type mistral \
  --prompts stage3_utility_prompts_v2.jsonl \
  --output stage3_results/mistral_utility_v2 \
  --methods permutation,signbit \
  --conditions clean,shuffle_05,lora_medium \
  --watermark-key YOUR_SECRET_KEY \
  --permutation-strength 0.002 \
  --verification-threshold 0.80 \
  --max-layers 8 \
  --enforce-quality-gate \
  --min-raw-choice-accuracy 0.60 \
  --min-raw-generation-valid-rate 0.60
```

## Main Components

### `src/permuteguard_large.py`
Contains the core PermuteGuard implementation, including:

- global seed control
- watermark configuration dataclass
- keyed metadata handling
- embedding and verification support
- large-model attack simulation helpers

### `src/open_source_model_loader.py`
Provides local model loading logic with support for:

- standard Transformer causal language models
- different load policies (`fp32_single`, `fp16_auto`, `int8_auto`)
- optional CPU offload
- multi-GPU memory mapping

### `stage2_weight_baselines.py`
Implements the sign-bit watermark baseline used in Stage 2 and Stage 3 comparisons.

### `stage2_benchmark_eval.py`
Handles benchmark normalization and evaluation for local JSONL task files.

## Expected Outputs

Typical outputs include:

- watermarked model metadata
- verification summaries
- benchmark evaluation JSON files
- utility-validation JSON files
- generated plots (`.png`)
- intermediate logs

The repository already includes historical outputs in:

- `results/`
- `stage2_results/`
- `stage3_results/`

These folders contain both stable and non-main-paper runs. For manuscript use, keep the evidence-selection policy consistent with the paper:

- **Stage 1**: final **v4** runs
- **Stage 2**: final **lora_v2** comparison runs plus the locked Mistral sensitivity runs
- **Stage 3**: only **gate-passing core runs** for main-paper claims

## Reproducibility Notes

For clean reproduction, document the following before release:

- Python version
- exact package versions
- CUDA version
- GPU type and memory
- exact local model checkpoint revisions
- secret key handling policy
- which result folders are main-paper evidence versus debug/supporting runs

## Models and Large Files

This archive intentionally does **not** include:

- local model checkpoints
- virtual environments
- downloaded Hugging Face caches
- large generated artifacts that are system-specific

Users must supply compatible local checkpoints before running the scripts.

## Citation

If you use this code in academic work, please cite the associated PermuteGuard manuscript.

## Disclaimer

This repository is a research artifact prepared around a manuscript workflow. Some folders contain exploratory, smoke, or supporting runs in addition to locked main-paper results. Users should verify which outputs are intended for formal reporting before reusing them in publications.
