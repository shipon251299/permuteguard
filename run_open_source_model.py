import argparse
import json
import math
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
    calculate_perplexity,
    set_global_seed,
)

EVAL_TEXTS = [
    "Researchers increasingly evaluate language models under realistic weight perturbations rather than single idealized settings.",
    "Ownership verification becomes difficult when a protected model is pruned, quantized, adapted, or partially reparameterized after deployment.",
    "A practical watermark should remain detectable without imposing a large and obvious quality penalty on normal language generation.",
    "Many industrial systems rely on compressed model variants, which means watermarking methods must tolerate imperfect deployment conditions.",
    "Robustness evaluation should distinguish between detectability and practical usefulness, because a broken model can still carry a detectable trace.",
    "Experimental results are more convincing when they are averaged across repeated trials instead of relying on a single favorable run.",
    "Perplexity is an imperfect but convenient utility metric for comparing whether attacks preserve the language modeling behavior of a system.",
    "Modern open-source models differ in scale, architecture, tokenizer design, and attention implementation, which can affect watermark stability.",
    "Pruning removes low-magnitude parameters, while quantization compresses numeric precision and noise injection perturbs weights stochastically.",
    "Weight shuffling is a harsher corruption that can preserve global statistics while severely changing local parameter structure.",
    "Fine-tuning style attacks are especially relevant because model owners and downstream users often continue training pretrained checkpoints.",
    "A strong experimental design should include compact baselines, medium models, and at least one large model to test scalability.",
    "Not every attack that preserves a detectable watermark also preserves utility, and that difference should be reported explicitly.",
    "False-positive resistance matters because a watermark detector must reject wrong keys and unrelated suspect models reliably.",
    "Model editing and parameter-efficient adaptation can alter only a small subset of weights, yet still change downstream behavior in nontrivial ways.",
    "Academic evaluation benefits from transparent code, repeatable attack settings, and plots that visualize both confidence and utility.",
    "Language models can appear stable under one metric while behaving very differently on unseen prompts, so evaluation sets should be diverse.",
    "A watermark that survives mild compression but fails under stronger restructuring may still be useful in practical forensic settings.",
    "Distributed loading across multiple GPUs can make large-model experiments feasible, but it also introduces implementation pitfalls.",
    "Careful engineering is required to separate experimental findings from artifacts caused by loaders, tokenizers, or numerical precision issues.",
    "Small models are useful sanity checks because their behavior is easier to debug and they run quickly enough for repeated ablations.",
    "Large models provide stronger evidence for scalability, especially when the selected watermarked layers are distributed across the network depth.",
    "When a utility metric changes in either direction too much, the attacked model should not automatically be considered practically preserved.",
    "Results are more believable when some attacks succeed, some partially succeed, and some clearly fail on utility or verification.",
]


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


def summarize(values: List[float]) -> Dict:
    clean = [float(v) for v in values if isinstance(v, (int, float))]
    if not clean:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": float(statistics.mean(clean)),
        "std": float(statistics.stdev(clean)) if len(clean) > 1 else 0.0,
        "min": float(min(clean)),
        "max": float(max(clean)),
        "n": len(clean),
    }


def utility_preserved(reference_ppl: float, attacked_ppl: float, reference_capped: bool, attacked_capped: bool, tolerance: float) -> bool:
    if reference_capped or attacked_capped:
        return False
    if not math.isfinite(reference_ppl) or not math.isfinite(attacked_ppl):
        return False
    if reference_ppl <= 0:
        return False
    delta_ratio = abs(attacked_ppl - reference_ppl) / reference_ppl
    return delta_ratio <= tolerance


def utility_delta_ratio(reference_ppl: float, attacked_ppl: float) -> float:
    if not math.isfinite(reference_ppl) or not math.isfinite(attacked_ppl) or reference_ppl <= 0:
        return float("nan")
    return float(abs(attacked_ppl - reference_ppl) / reference_ppl)


def default_attack_grid(model_type: str) -> Dict[str, Dict]:
    if model_type in {"mistral", "falcon", "bloom", "opt"}:
        return {
            "fine_tuning_light": {"attack_name": "constrained_adaptation", "lr": 2e-7, "steps": 2},
            "fine_tuning_medium": {"attack_name": "constrained_adaptation", "lr": 5e-7, "steps": 4},
            "pruning_10": {"attack_name": "pruning", "sparsity": 0.10},
            "pruning_20": {"attack_name": "pruning", "sparsity": 0.20},
            "quantization_8bit": {"attack_name": "quantization", "bits": 8},
            "noise_0.002": {"attack_name": "noise_injection", "noise_factor": 0.002},
            "noise_0.01": {"attack_name": "noise_injection", "noise_factor": 0.01},
            "shuffle_05": {"attack_name": "weight_shuffling", "shuffle_rate": 0.05},
            "shuffle_10": {"attack_name": "weight_shuffling", "shuffle_rate": 0.10},
        }

    return {
        "adaptation_light": {"attack_name": "constrained_adaptation", "lr": 1e-6, "steps": 4},
        "adaptation_medium": {"attack_name": "constrained_adaptation", "lr": 2e-6, "steps": 8},
        "pruning_10": {"attack_name": "pruning", "sparsity": 0.10},
        "pruning_20": {"attack_name": "pruning", "sparsity": 0.20},
        "quantization_8bit": {"attack_name": "quantization", "bits": 8},
        "noise_0.002": {"attack_name": "noise_injection", "noise_factor": 0.002},
        "noise_0.01": {"attack_name": "noise_injection", "noise_factor": 0.01},
        "shuffle_05": {"attack_name": "weight_shuffling", "shuffle_rate": 0.05},
        "shuffle_10": {"attack_name": "weight_shuffling", "shuffle_rate": 0.10},
    }


def create_summary_and_plots(results: Dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    rows = []
    for attack_name, payload in results["attack_results"].items():
        rows.append(
            {
                "attack": attack_name,
                "confidence_mean": payload["confidence"]["mean"],
                "confidence_std": payload["confidence"]["std"],
                "perplexity_mean": payload["perplexity"]["mean"],
                "utility_delta_mean": payload["utility_delta_ratio"]["mean"],
                "verified_rate": payload["verified_rate"],
                "utility_ok_rate": payload["utility_ok_rate"],
                "practical_robust_rate": payload["practical_robust_rate"],
                "failure_rate": payload["failure_rate"],
                "perplexity_capped_rate": payload["perplexity_capped_rate"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "attack_summary.csv"), index=False)

    threshold = results["watermark_config"]["verification_threshold"]

    plt.figure(figsize=(11, 6))
    plt.bar(df["attack"], df["confidence_mean"], yerr=df["confidence_std"], capsize=4)
    plt.axhline(threshold, linestyle="--")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Mean watermark confidence")
    plt.title("Watermark confidence by attack")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "confidence_by_attack.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.bar(df["attack"], df["perplexity_mean"])
    plt.yscale("log")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Mean perplexity (log scale)")
    plt.title("Perplexity by attack")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "perplexity_by_attack_log.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.bar(df["attack"], df["practical_robust_rate"], label="Practical robust")
    plt.bar(df["attack"], df["utility_ok_rate"], alpha=0.55, label="Utility preserved")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Rate")
    plt.ylim(0, 1.05)
    plt.title("Robustness rates by attack")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "robustness_rates.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(df["confidence_mean"], df["utility_delta_mean"])
    for _, row in df.iterrows():
        plt.annotate(row["attack"], (row["confidence_mean"], row["utility_delta_mean"]), fontsize=8)
    plt.xlabel("Mean confidence")
    plt.ylabel("Mean relative utility drift")
    plt.title("Confidence vs utility drift")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "confidence_vs_utility_drift.png"), dpi=300, bbox_inches="tight")
    plt.close()


def run_experiment(args):
    set_global_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    if args.model_type == "gpt2" and args.materialized_dtype == "float16":
        print("[INFO] Overriding GPT-2 materialized dtype from float16 to float32 for stability.")
        args.materialized_dtype = "float32"

    model, tokenizer = OpenSourceModelLoader.load_model(
        model_path=args.model,
        model_type=args.model_type,
        load_policy=args.load_policy,
        max_gpu_memory_gib=args.max_gpu_memory_gib,
        cpu_memory_gib=args.cpu_memory_gib,
        enable_cpu_offload=not args.disable_cpu_offload,
    )
    print("tokenizer length:", len(tokenizer))
    print("model vocab_size:", getattr(model.config, "vocab_size", None))

    raw_baseline_ppl, raw_baseline_capped, raw_baseline_status = calculate_perplexity(
        model, tokenizer, EVAL_TEXTS, max_length=args.max_length, ppl_cap=args.perplexity_cap
    )

    pg = PermuteGuardLarge(
        LargeWatermarkConfig(
            watermark_key=args.watermark_key or f"PG::{args.model_type}::{os.path.basename(args.model)}",
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

    planned_layers = [name for name, _ in pg.identify_candidate_layers(model)]
    pg.materialize_target_modules(model, planned_layers)

    reference_ppl, reference_capped, reference_status = calculate_perplexity(
        model, tokenizer, EVAL_TEXTS, max_length=args.max_length, ppl_cap=args.perplexity_cap
    )

    model, metadata = pg.embed_watermark(model, args.model, selected_layers=planned_layers, already_materialized=True)
    selected_layers = metadata["selected_layers"]

    watermarked_ppl, watermarked_capped, watermarked_status = calculate_perplexity(
        model, tokenizer, EVAL_TEXTS, max_length=args.max_length, ppl_cap=args.perplexity_cap
    )

    initial_verification = pg.verify_watermark(model, args.model, watermark_key=pg.config.watermark_key)
    false_positive_check = pg.verify_watermark(model, args.model, watermark_key="definitely_the_wrong_key")

    clean_snapshot = pg.snapshot_selected_layers(model, selected_layers)
    simulator = AttackSimulatorLarge(EVAL_TEXTS, max_length=args.max_length)
    attack_grid = default_attack_grid(args.model_type)

    attack_results = {}

    for attack_label, attack_spec in attack_grid.items():
        print(f"\n=== Running attack: {attack_label} ===")

        trial_confidences = []
        trial_verified = []
        trial_ppl = []
        trial_utility_ok = []
        trial_practical = []
        trial_delta = []
        trial_failures = 0
        trial_capped = 0
        per_trial = []

        for trial in range(args.trials):
            print(f"  Trial {trial + 1}/{args.trials}")
            pg.restore_selected_layers(model, clean_snapshot)

            verify = {}
            conf = 0.0
            verified_flag = False
            ppl = args.perplexity_cap
            ppl_capped = True
            ppl_status = "attack_failed"
            utility_ok = False
            practical_robust = False
            attack_success = False
            delta_ratio = float("nan")

            try:
                attack_name = attack_spec["attack_name"]
                kwargs = dict(attack_spec)
                kwargs.pop("attack_name")

                if attack_name == "constrained_adaptation":
                    model = simulator.constrained_adaptation(model, tokenizer=tokenizer, layer_names=selected_layers, **kwargs)
                elif attack_name == "pruning":
                    model = simulator.pruning(model, layer_names=selected_layers, **kwargs)
                elif attack_name == "quantization":
                    model = simulator.quantization(model, layer_names=selected_layers, **kwargs)
                elif attack_name == "noise_injection":
                    model = simulator.noise_injection(model, layer_names=selected_layers, **kwargs)
                elif attack_name == "weight_shuffling":
                    model = simulator.weight_shuffling(model, layer_names=selected_layers, **kwargs)
                else:
                    raise ValueError(f"Unknown attack_name: {attack_name}")

                verify = pg.verify_watermark(model, args.model, watermark_key=pg.config.watermark_key)
                conf = float(verify["confidence"])
                verified_flag = bool(verify["verified"])

                ppl, ppl_capped, ppl_status = calculate_perplexity(
                    model, tokenizer, EVAL_TEXTS, max_length=args.max_length, ppl_cap=args.perplexity_cap
                )

                delta_ratio = utility_delta_ratio(reference_ppl, ppl)
                utility_ok = utility_preserved(reference_ppl, ppl, reference_capped, ppl_capped, args.utility_tolerance)
                practical_robust = verified_flag and utility_ok
                attack_success = True

            except Exception as exc:
                trial_failures += 1
                verify = {"error": str(exc)}

            if ppl_capped:
                trial_capped += 1

            print(
                f"    confidence={conf:.6f}, verified={verified_flag}, perplexity={ppl:.4f}, "
                f"utility_ok={utility_ok}, practical_robust={practical_robust}, status={ppl_status}"
            )

            trial_confidences.append(float(conf))
            trial_verified.append(bool(verified_flag))
            trial_ppl.append(float(ppl))
            trial_utility_ok.append(bool(utility_ok))
            trial_practical.append(bool(practical_robust))
            trial_delta.append(float(delta_ratio) if math.isfinite(delta_ratio) else float("nan"))
            per_trial.append(
                {
                    "trial": trial + 1,
                    "confidence": float(conf),
                    "verified": bool(verified_flag),
                    "perplexity": float(ppl),
                    "perplexity_capped": bool(ppl_capped),
                    "perplexity_status": ppl_status,
                    "utility_ok": bool(utility_ok),
                    "practical_robust": bool(practical_robust),
                    "utility_delta_ratio": float(delta_ratio) if math.isfinite(delta_ratio) else None,
                    "attack_success": bool(attack_success),
                    "error": verify.get("error"),
                }
            )

            cleanup()

        attack_results[attack_label] = {
            "attack": attack_spec,
            "confidence": summarize(trial_confidences),
            "verified_rate": float(sum(trial_verified) / len(trial_verified)),
            "perplexity": summarize(trial_ppl),
            "utility_delta_ratio": summarize([v for v in trial_delta if math.isfinite(v)]),
            "utility_ok_rate": float(sum(trial_utility_ok) / len(trial_utility_ok)),
            "practical_robust_rate": float(sum(trial_practical) / len(trial_practical)),
            "failure_rate": float(trial_failures / args.trials),
            "perplexity_capped_rate": float(trial_capped / args.trials),
            "trials": per_trial,
        }

    try:
        total_parameters = int(sum(p.numel() for p in model.parameters()))
    except Exception:
        total_parameters = None

    relative_change = utility_delta_ratio(reference_ppl, watermarked_ppl)
    utility_ok_after_watermark = utility_preserved(reference_ppl, watermarked_ppl, reference_capped, watermarked_capped, args.utility_tolerance)

    results = {
        "experiment_info": {
            "model": args.model,
            "model_type": args.model_type,
            "timestamp": datetime.now().isoformat(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "total_parameters": total_parameters,
            "trials_per_attack": args.trials,
            "load_policy": args.load_policy,
        },
        "watermark_config": {
            "permutation_strength": args.permutation_strength,
            "verification_threshold": args.verification_threshold,
            "fingerprint_k": args.fingerprint_k,
            "max_layers": args.max_layers,
            "min_layer_size": args.min_layer_size,
            "materialized_dtype": args.materialized_dtype,
        },
        "utility": {
            "raw_baseline_perplexity": float(raw_baseline_ppl),
            "raw_baseline_perplexity_capped": bool(raw_baseline_capped),
            "raw_baseline_status": raw_baseline_status,
            "reference_perplexity": float(reference_ppl),
            "reference_perplexity_capped": bool(reference_capped),
            "reference_status": reference_status,
            "utility_reference_perplexity": float(reference_ppl),
            "utility_reference_perplexity_capped": bool(reference_capped),
            "utility_reference_status": reference_status,
            "watermarked_perplexity": float(watermarked_ppl),
            "watermarked_perplexity_capped": bool(watermarked_capped),
            "watermarked_status": watermarked_status,
            "relative_change": float(relative_change) if math.isfinite(relative_change) else None,
            "utility_tolerance": args.utility_tolerance,
            "utility_ok_after_watermark": bool(utility_ok_after_watermark),
        },
        "selected_layers": metadata["selected_layer_info"],
        "initial_verification": initial_verification,
        "false_positive_check": false_positive_check,
        "attack_results": attack_results,
    }

    json_path = os.path.join(args.output, f"results_{os.path.basename(args.model)}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    create_summary_and_plots(results, args.output)

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {os.path.join(args.output, 'attack_summary.csv')}")
    print(f"Saved plots in: {os.path.join(args.output, 'plots')}")
    cleanup()


def build_parser():
    parser = argparse.ArgumentParser(description="Run local open-source LLM watermark experiments.")
    parser.add_argument("--model", required=True, help="Local model directory")
    parser.add_argument("--model-type", required=True, choices=["gpt2", "tinyllama", "mistral", "falcon", "bloom", "opt"])
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--trials", type=int, default=3)

    parser.add_argument("--load-policy", default="auto", choices=["auto", "fp32_single", "fp16_auto", "int8_auto"])
    parser.add_argument("--max-gpu-memory-gib", type=int, default=18)
    parser.add_argument("--cpu-memory-gib", type=int, default=120)
    parser.add_argument("--disable-cpu-offload", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--watermark-key", default=None)
    parser.add_argument("--permutation-strength", type=float, default=0.003)
    parser.add_argument("--fingerprint-k", type=int, default=4096)
    parser.add_argument("--verification-threshold", type=float, default=0.80)
    parser.add_argument("--max-layers", type=int, default=8)
    parser.add_argument("--min-layer-size", type=int, default=50000)
    parser.add_argument("--include-embeddings", action="store_true")
    parser.add_argument("--materialized-dtype", choices=["float16", "float32"], default="float16")

    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--perplexity-cap", type=float, default=1000000.0)
    parser.add_argument("--utility-tolerance", type=float, default=0.20)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_experiment(args)
