
import argparse
import copy
import json
import os
import re
import statistics
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.open_source_model_loader import OpenSourceModelLoader
from src.permuteguard_large import (
    AttackSimulatorLarge,
    LargeWatermarkConfig,
    PermuteGuardLarge,
    get_input_device,
    set_global_seed,
)
from stage2_weight_baselines import SignBitConfig, SignBitWatermark


EVAL_TEXTS = [
    "Researchers increasingly evaluate language models under realistic weight perturbations rather than single idealized settings.",
    "Ownership verification becomes difficult when a protected model is pruned, quantized, adapted, or partially reparameterized after deployment.",
    "A practical watermark should remain detectable without imposing a large and obvious quality penalty on normal language generation.",
    "Robustness evaluation should distinguish between detectability and practical usefulness, because a broken model can still carry a detectable trace.",
    "Results are more believable when some attacks succeed, some partially succeed, and some clearly fail on utility or verification.",
]


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"^[^a-z0-9+-]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def extract_first_answer(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    first_line = text.splitlines()[0].strip()
    return first_line if first_line else text[:80].strip()


def match_label(row: Dict, response: str) -> Optional[str]:
    answer = extract_first_answer(response)
    answer_norm = normalize_text(answer)

    if row.get("category") == "arithmetic":
        m = re.search(r"-?\d+", answer_norm)
        if m:
            return m.group(0)
        return None

    for label in row["valid_labels"]:
        aliases = row.get("aliases", {}).get(label, [label.lower()])
        for alias in aliases:
            alias_norm = normalize_text(alias)
            if not alias_norm:
                continue
            if answer_norm == alias_norm:
                return label
            if answer_norm.startswith(alias_norm + " "):
                return label
            if answer_norm.startswith(alias_norm + "."):
                return label
            if answer_norm.startswith(alias_norm + ":"):
                return label
            if answer_norm.startswith("answer " + alias_norm):
                return label
            if answer_norm.startswith("the answer is " + alias_norm):
                return label
    return None


def summarize(values: List[float]) -> Dict:
    vals = [float(v) for v in values]
    return {
        "mean": float(statistics.mean(vals)) if vals else None,
        "std": float(statistics.stdev(vals)) if len(vals) > 1 else (0.0 if vals else None),
        "min": float(min(vals)) if vals else None,
        "max": float(max(vals)) if vals else None,
        "n": len(vals),
    }


def cleanup():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def attack_grid(model_type: str) -> Dict[str, Dict]:
    if model_type in {"mistral", "falcon", "bloom", "opt"}:
        return {
            "clean": {"attack_name": "none"},
            "shuffle_05": {"attack_name": "weight_shuffling", "shuffle_rate": 0.05},
            "lora_medium": {
                "attack_name": "lora_adaptation",
                "rank": 8,
                "alpha": 16,
                "dropout": 0.0,
                "lr": 1e-4,
                "steps": 500,
            },
        }
    return {
        "clean": {"attack_name": "none"},
        "shuffle_05": {"attack_name": "weight_shuffling", "shuffle_rate": 0.05},
        "lora_medium": {
            "attack_name": "lora_adaptation",
            "rank": 8,
            "alpha": 16,
            "dropout": 0.0,
            "lr": 1e-4,
            "steps": 300,
        },
    }


def build_watermarker(args, method: str, output_dir: str):
    if method == "permutation":
        return PermuteGuardLarge(
            LargeWatermarkConfig(
                watermark_key=args.watermark_key or f"PG::{args.model_type}::{os.path.basename(args.model)}",
                metadata_dir=output_dir,
                permutation_strength=args.permutation_strength,
                fingerprint_k=args.fingerprint_k,
                verification_threshold=args.verification_threshold,
                max_layers=args.max_layers,
                min_layer_size=args.min_layer_size,
                exclude_embeddings=not args.include_embeddings,
                materialized_dtype=args.materialized_dtype,
            )
        )
    if method == "signbit":
        return SignBitWatermark(
            SignBitConfig(
                watermark_key=args.watermark_key or f"SB::{args.model_type}::{os.path.basename(args.model)}",
                metadata_dir=output_dir,
                bit_fraction=args.permutation_strength,
                verification_threshold=args.verification_threshold,
                fingerprint_k=args.fingerprint_k,
            )
        )
    raise ValueError(f"Unknown method: {method}")


def verify_method(wm, model, model_id, watermark_key):
    if isinstance(wm, PermuteGuardLarge):
        return wm.verify_watermark(model, model_id, watermark_key=watermark_key)
    return wm.verify(model, model_id, watermark_key=watermark_key)


def embed_method(wm, model, model_id, selected_layers):
    if isinstance(wm, PermuteGuardLarge):
        return wm.embed_watermark(model, model_id, selected_layers=selected_layers, already_materialized=True)
    return wm.embed(model, model_id, selected_layers=selected_layers)


def generate_text(model, tokenizer, prompt: str, max_input_length: int = 256, max_new_tokens: int = 8) -> str:
    device = get_input_device(model)
    batch = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
        add_special_tokens=True,
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id,
    )
    with torch.no_grad():
        out = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            **gen_kwargs,
        )
    gen_ids = out[0, batch["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def evaluate_prompts(model, tokenizer, prompts: List[Dict], max_input_length: int, max_new_tokens: int, progress_every: int = 10) -> Dict:
    details = []
    correct = 0
    valid = 0
    nonempty = 0

    by_category_counts = {}
    by_category_correct = {}
    by_category_valid = {}

    for idx, row in enumerate(prompts, start=1):
        response = generate_text(model, tokenizer, row["prompt"], max_input_length=max_input_length, max_new_tokens=max_new_tokens)
        pred = match_label(row, response)
        is_nonempty = bool(response.strip())
        is_valid = pred is not None
        is_correct = pred == row["gold"]

        nonempty += int(is_nonempty)
        valid += int(is_valid)
        correct += int(is_correct)

        cat = row.get("category", "unknown")
        by_category_counts[cat] = by_category_counts.get(cat, 0) + 1
        by_category_valid[cat] = by_category_valid.get(cat, 0) + int(is_valid)
        by_category_correct[cat] = by_category_correct.get(cat, 0) + int(is_correct)

        details.append(
            {
                "id": row["id"],
                "category": cat,
                "gold": row["gold"],
                "pred": pred,
                "valid": bool(is_valid),
                "correct": bool(is_correct),
                "response": response,
            }
        )

        if progress_every > 0 and idx % progress_every == 0:
            print(f"  evaluated {idx}/{len(prompts)} prompts")

    per_category = {}
    for cat, n in by_category_counts.items():
        per_category[cat] = {
            "n": n,
            "valid_rate": by_category_valid[cat] / max(n, 1),
            "accuracy_over_all": by_category_correct[cat] / max(n, 1),
            "accuracy_valid_only": by_category_correct[cat] / max(by_category_valid[cat], 1),
        }

    return {
        "n": len(prompts),
        "nonempty_rate": nonempty / max(len(prompts), 1),
        "valid_rate": valid / max(len(prompts), 1),
        "accuracy_over_all": correct / max(len(prompts), 1),
        "accuracy_valid_only": correct / max(valid, 1),
        "invalid_examples": int(len(prompts) - valid),
        "per_category": per_category,
        "details": details,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", required=True, choices=["gpt2", "tinyllama", "mistral", "falcon", "bloom", "opt"])
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--methods", default="permutation,signbit")
    ap.add_argument("--conditions", default="clean,shuffle_05,lora_medium")
    ap.add_argument("--load-policy", default="auto", choices=["auto", "fp32_single", "fp16_auto", "int8_auto"])
    ap.add_argument("--max-gpu-memory-gib", type=int, default=18)
    ap.add_argument("--cpu-memory-gib", type=int, default=120)
    ap.add_argument("--disable-cpu-offload", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--watermark-key", default=None)
    ap.add_argument("--permutation-strength", type=float, default=0.003)
    ap.add_argument("--fingerprint-k", type=int, default=4096)
    ap.add_argument("--verification-threshold", type=float, default=0.80)
    ap.add_argument("--max-layers", type=int, default=8)
    ap.add_argument("--min-layer-size", type=int, default=50000)
    ap.add_argument("--include-embeddings", action="store_true")
    ap.add_argument("--materialized-dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--max-input-length", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--max-prompts", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=10)
    args = ap.parse_args()

    set_global_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    prompts = read_jsonl(args.prompts)
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    condition_names = [x.strip() for x in args.conditions.split(",") if x.strip()]
    grid = attack_grid(args.model_type)

    model, tokenizer = OpenSourceModelLoader.load_model(
        model_path=args.model,
        model_type=args.model_type,
        load_policy=args.load_policy,
        max_gpu_memory_gib=args.max_gpu_memory_gib,
        cpu_memory_gib=args.cpu_memory_gib,
        enable_cpu_offload=not args.disable_cpu_offload,
    )

    helper_pg = PermuteGuardLarge(
        LargeWatermarkConfig(
            watermark_key="selector",
            metadata_dir=args.output,
            permutation_strength=args.permutation_strength,
            fingerprint_k=args.fingerprint_k,
            verification_threshold=args.verification_threshold,
            max_layers=args.max_layers,
            min_layer_size=args.min_layer_size,
            exclude_embeddings=not args.include_embeddings,
            materialized_dtype=args.materialized_dtype,
        )
    )
    selected_layers = [n for n, _ in helper_pg.identify_candidate_layers(model)]
    helper_pg.materialize_target_modules(model, selected_layers)

    raw_eval = evaluate_prompts(
        model, tokenizer, prompts,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        progress_every=args.progress_every,
    )
    clean_snapshot = helper_pg.snapshot_selected_layers(model, selected_layers)

    simulator = AttackSimulatorLarge(EVAL_TEXTS, max_length=128)
    results = {
        "experiment_info": {
            "model": args.model,
            "model_type": args.model_type,
            "timestamp": datetime.now().isoformat(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "load_policy": args.load_policy,
            "num_prompts": len(prompts),
        },
        "selected_layers": selected_layers,
        "raw_model": raw_eval,
        "methods": {},
    }

    for method in methods:
        helper_pg.restore_selected_layers(model, clean_snapshot)
        method_dir = os.path.join(args.output, method)
        os.makedirs(method_dir, exist_ok=True)

        wm = build_watermarker(args, method, method_dir)
        model, _ = embed_method(wm, model, args.model, selected_layers)
        wm_key = getattr(wm, "config").watermark_key

        clean_eval = evaluate_prompts(
            model, tokenizer, prompts,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            progress_every=args.progress_every,
        )
        clean_verify = verify_method(wm, model, args.model, wm_key)
        wm_snapshot = helper_pg.snapshot_selected_layers(model, selected_layers)

        method_out = {
            "clean": {
                "utility": clean_eval,
                "verification": clean_verify,
            },
            "conditions": {},
        }

        for condition_name in condition_names:
            if condition_name == "clean":
                method_out["conditions"]["clean"] = {
                    "utility": clean_eval,
                    "verification": clean_verify,
                }
                continue

            if condition_name not in grid:
                raise ValueError(f"Unknown condition: {condition_name}")

            helper_pg.restore_selected_layers(model, wm_snapshot)
            spec = copy.deepcopy(grid[condition_name])
            attack_name = spec.pop("attack_name")

            if attack_name == "weight_shuffling":
                model = simulator.weight_shuffling(model, layer_names=selected_layers, **spec)
            elif attack_name == "lora_adaptation":
                model = simulator.lora_adaptation(model, tokenizer=tokenizer, layer_names=selected_layers, **spec)
            elif attack_name == "none":
                pass
            else:
                raise ValueError(f"Unsupported attack function: {attack_name}")

            cond_eval = evaluate_prompts(
                model, tokenizer, prompts,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
                progress_every=args.progress_every,
            )
            cond_verify = verify_method(wm, model, args.model, wm_key)

            method_out["conditions"][condition_name] = {
                "utility": cond_eval,
                "verification": cond_verify,
            }
            cleanup()

        results["methods"][method] = method_out

    json_path = os.path.join(args.output, f"stage3_utility_{os.path.basename(args.model)}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    rows = []
    for method, payload in results["methods"].items():
        for cond, cpay in payload["conditions"].items():
            rows.append({
                "method": method,
                "condition": cond,
                "accuracy_over_all": cpay["utility"]["accuracy_over_all"],
                "accuracy_valid_only": cpay["utility"]["accuracy_valid_only"],
                "valid_rate": cpay["utility"]["valid_rate"],
                "nonempty_rate": cpay["utility"]["nonempty_rate"],
                "invalid_examples": cpay["utility"]["invalid_examples"],
                "verification_confidence": cpay["verification"]["confidence"],
            })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output, "stage3_utility_summary.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5))
    for method in df["method"].unique():
        sub = df[df["method"] == method]
        plt.plot(sub["condition"], sub["accuracy_over_all"], marker="o", label=method)
    plt.ylabel("Accuracy over all prompts")
    plt.title("Stage 3 utility validation")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(args.output, "stage3_utility_accuracy.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
