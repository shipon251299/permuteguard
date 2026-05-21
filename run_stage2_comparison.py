import argparse
import copy
import json
import os
import statistics
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.open_source_model_loader import OpenSourceModelLoader
from src.permuteguard_large import (
    AttackSimulatorLarge,
    LargeWatermarkConfig,
    PermuteGuardLarge,
    set_global_seed,
)
from stage2_benchmark_eval import (
    normalize_row,
    read_jsonl,
    select_stable_subset_generation,
    evaluate_task_files_generation,
)
from stage2_weight_baselines import SignBitConfig, SignBitWatermark


EVAL_TEXTS = [
    "Researchers increasingly evaluate language models under realistic weight perturbations rather than single idealized settings.",
    "Ownership verification becomes difficult when a protected model is pruned, quantized, adapted, or partially reparameterized after deployment.",
    "A practical watermark should remain detectable without imposing a large and obvious quality penalty on normal language generation.",
    "Robustness evaluation should distinguish between detectability and practical usefulness, because a broken model can still carry a detectable trace.",
    "Results are more believable when some attacks succeed, some partially succeed, and some clearly fail on utility or verification.",
]


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
            "none": {"attack_name": "none"},
            "fine_tuning_light": {
                "attack_name": "constrained_adaptation",
                "lr": 2e-7,
                "steps": 2,
            },
            "lora_light": {
                "attack_name": "lora_adaptation",
                "rank": 8,
                "alpha": 16,
                "dropout": 0.0,
                "lr": 5e-5,
                "steps": 200,
            },
            "lora_medium": {
                "attack_name": "lora_adaptation",
                "rank": 8,
                "alpha": 16,
                "dropout": 0.0,
                "lr": 1e-4,
                "steps": 500,
            },
            "pruning_10": {"attack_name": "pruning", "sparsity": 0.10},
            "quantization_8bit": {"attack_name": "quantization", "bits": 8},
            "shuffle_05": {"attack_name": "weight_shuffling", "shuffle_rate": 0.05},
        }

    return {
        "none": {"attack_name": "none"},
        "adaptation_medium": {
            "attack_name": "constrained_adaptation",
            "lr": 2e-6,
            "steps": 8,
        },
        "lora_light": {
            "attack_name": "lora_adaptation",
            "rank": 8,
            "alpha": 16,
            "dropout": 0.0,
            "lr": 5e-5,
            "steps": 200,
        },
        "lora_medium": {
            "attack_name": "lora_adaptation",
            "rank": 8,
            "alpha": 16,
            "dropout": 0.0,
            "lr": 1e-4,
            "steps": 300,
        },
        "pruning_10": {"attack_name": "pruning", "sparsity": 0.10},
        "quantization_8bit": {"attack_name": "quantization", "bits": 8},
        "shuffle_05": {"attack_name": "weight_shuffling", "shuffle_rate": 0.05},
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", required=True, choices=["gpt2", "tinyllama", "mistral", "falcon", "bloom", "opt"])
    ap.add_argument("--task-files", required=True, help="Comma-separated local JSONL benchmark files")
    ap.add_argument("--output", required=True)
    ap.add_argument("--methods", default="permutation,signbit")
    ap.add_argument("--attacks", default="none,pruning_10,quantization_8bit,shuffle_05")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--max-examples", type=int, default=50)
    ap.add_argument("--max-scan", type=int, default=300)
    ap.add_argument("--mcq-max-length", type=int, default=384)
    ap.add_argument("--gen-max-new-tokens", type=int, default=3)
    ap.add_argument("--progress-every", type=int, default=10)

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
    args = ap.parse_args()

    set_global_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)
    task_files = [x.strip() for x in args.task_files.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    attack_names = [x.strip() for x in args.attacks.split(",") if x.strip()]
    full_grid = attack_grid(args.model_type)

    model, tokenizer = OpenSourceModelLoader.load_model(
        model_path=args.model,
        model_type=args.model_type,
        load_policy=args.load_policy,
        max_gpu_memory_gib=args.max_gpu_memory_gib,
        cpu_memory_gib=args.cpu_memory_gib,
        enable_cpu_offload=not args.disable_cpu_offload,
    )

    raw_task_rows = {}
    stable_subset_meta = {}
    for path in task_files:
        task_name = os.path.splitext(os.path.basename(path))[0]
        print(f"\nSelecting stable subset for task: {task_name}")
        rows = [normalize_row(r) for r in read_jsonl(path)]
        out = select_stable_subset_generation(
            model,
            tokenizer,
            task_name,
            rows,
            target_n=args.max_examples,
            max_scan=args.max_scan,
            max_input_length=args.mcq_max_length,
            max_new_tokens=args.gen_max_new_tokens,
            progress_every=args.progress_every,
        )
        raw_task_rows[task_name] = out["selected_rows"]
        stable_subset_meta[task_name] = out["meta"]

    raw_task_eval = evaluate_task_files_generation(
        model,
        tokenizer,
        raw_task_rows,
        max_input_length=args.mcq_max_length,
        max_new_tokens=args.gen_max_new_tokens,
        progress_every=args.progress_every,
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

    reference_task_eval = evaluate_task_files_generation(
        model,
        tokenizer,
        raw_task_rows,
        max_input_length=args.mcq_max_length,
        max_new_tokens=args.gen_max_new_tokens,
        progress_every=args.progress_every,
    )

    simulator = AttackSimulatorLarge(EVAL_TEXTS, max_length=128)
    results = {
        "experiment_info": {
            "model": args.model,
            "model_type": args.model_type,
            "timestamp": datetime.now().isoformat(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "trials_per_attack": args.trials,
            "load_policy": args.load_policy,
            "task_files": task_files,
        },
        "stable_subset": stable_subset_meta,
        "raw_model": raw_task_eval,
        "reference_model": reference_task_eval,
        "selected_layers": selected_layers,
        "methods": {},
    }

    clean_snapshot = helper_pg.snapshot_selected_layers(model, selected_layers)

    for method in methods:
        method_dir = os.path.join(args.output, method)
        os.makedirs(method_dir, exist_ok=True)

        helper_pg.restore_selected_layers(model, clean_snapshot)
        wm = build_watermarker(args, method, method_dir)
        model, metadata = embed_method(wm, model, args.model, selected_layers)
        wm_key = getattr(wm, "config").watermark_key

        clean_verify = verify_method(wm, model, args.model, wm_key)
        clean_task_eval = evaluate_task_files_generation(
            model,
            tokenizer,
            raw_task_rows,
            max_input_length=args.mcq_max_length,
            max_new_tokens=args.gen_max_new_tokens,
            progress_every=args.progress_every,
        )
        wm_snapshot = helper_pg.snapshot_selected_layers(model, selected_layers)

        method_out = {
            "clean": {
                "verification": clean_verify,
                "macro_accuracy_over_subset": clean_task_eval["macro_accuracy_over_subset"],
                "macro_accuracy_valid_only": clean_task_eval["macro_accuracy_valid_only"],
                "task_eval": clean_task_eval,
            },
            "attacks": {},
        }

        for attack_name in attack_names:
            if attack_name not in full_grid:
                raise ValueError(f"Unknown attack: {attack_name}")
            spec = copy.deepcopy(full_grid[attack_name])

            acc_subset, acc_valid, invalids, confs, per_trial = [], [], [], [], []
            for trial in range(args.trials):
                helper_pg.restore_selected_layers(model, wm_snapshot)
                verify = {}
                acc_s = 0.0
                acc_v = 0.0
                invalid_n = 0
                success = False
                error = None

                try:
                    attack_fn = spec["attack_name"]
                    kwargs = dict(spec)
                    kwargs.pop("attack_name")

                    if attack_fn == "none":
                        pass
                    elif attack_fn == "constrained_adaptation":
                        model = simulator.constrained_adaptation(
                            model,
                            tokenizer=tokenizer,
                            layer_names=selected_layers,
                            **kwargs,
                        )
                    elif attack_fn == "lora_adaptation":
                        model = simulator.lora_adaptation(
                            model,
                            tokenizer=tokenizer,
                            layer_names=selected_layers,
                            **kwargs,
                        )
                    elif attack_fn == "pruning":
                        model = simulator.pruning(model, layer_names=selected_layers, **kwargs)
                    elif attack_fn == "quantization":
                        model = simulator.quantization(model, layer_names=selected_layers, **kwargs)
                    elif attack_fn == "noise_injection":
                        model = simulator.noise_injection(model, layer_names=selected_layers, **kwargs)
                    elif attack_fn == "weight_shuffling":
                        model = simulator.weight_shuffling(model, layer_names=selected_layers, **kwargs)
                    else:
                        raise ValueError(f"Unsupported attack function {attack_fn}")

                    verify = verify_method(wm, model, args.model, wm_key)
                    t_eval = evaluate_task_files_generation(
                        model,
                        tokenizer,
                        raw_task_rows,
                        max_input_length=args.mcq_max_length,
                        max_new_tokens=args.gen_max_new_tokens,
                        progress_every=args.progress_every,
                    )
                    acc_s = t_eval["macro_accuracy_over_subset"]
                    acc_v = t_eval["macro_accuracy_valid_only"]
                    invalid_n = int(sum(x["invalid_examples"] for x in t_eval["tasks"].values()))
                    success = True

                except Exception as exc:
                    error = str(exc)

                confs.append(float(verify.get("confidence", 0.0)))
                acc_subset.append(float(acc_s))
                acc_valid.append(float(acc_v))
                invalids.append(float(invalid_n))
                per_trial.append(
                    {
                        "trial": trial + 1,
                        "confidence": float(verify.get("confidence", 0.0)),
                        "verified": bool(verify.get("verified", False)),
                        "macro_accuracy_over_subset": float(acc_s),
                        "macro_accuracy_valid_only": float(acc_v),
                        "invalid_examples": int(invalid_n),
                        "success": success,
                        "error": error,
                    }
                )
                cleanup()

            method_out["attacks"][attack_name] = {
                "confidence": summarize(confs),
                "macro_accuracy_over_subset": summarize(acc_subset),
                "macro_accuracy_valid_only": summarize(acc_valid),
                "invalid_examples": summarize(invalids),
                "trials": per_trial,
            }

        results["methods"][method] = method_out

    out_json = os.path.join(args.output, f"stage2_comparison_{os.path.basename(args.model)}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    rows = []
    for method, payload in results["methods"].items():
        clean_acc = payload["clean"]["macro_accuracy_over_subset"]
        clean_conf = payload["clean"]["verification"]["confidence"]
        for attack_name, apayload in payload["attacks"].items():
            rows.append(
                {
                    "method": method,
                    "attack": attack_name,
                    "clean_macro_accuracy_over_subset": clean_acc,
                    "clean_confidence": clean_conf,
                    "attack_macro_accuracy_over_subset_mean": apayload["macro_accuracy_over_subset"]["mean"],
                    "attack_macro_accuracy_valid_only_mean": apayload["macro_accuracy_valid_only"]["mean"],
                    "attack_confidence_mean": apayload["confidence"]["mean"],
                    "invalid_examples_mean": apayload["invalid_examples"]["mean"],
                }
            )
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output, "stage2_summary.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    for method in df["method"].unique():
        sub = df[df["method"] == method]
        plt.plot(sub["attack"], sub["attack_macro_accuracy_over_subset_mean"], marker="o", label=method)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Accuracy over fixed subset")
    plt.title("Task accuracy by method and attack")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "stage2_accuracy_plot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {out_json}")
    print(f"Saved {csv_path}")
    print(f"Saved {os.path.join(args.output, 'stage2_accuracy_plot.png')}")


if __name__ == "__main__":
    main()
