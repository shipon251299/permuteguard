import argparse
import copy
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

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


def maybe_chat_wrap(tokenizer, prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template or not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    try:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt


def normalize_text(s: str) -> str:
    s = s.upper()
    s = s.replace("\r", " ").replace("\n", " ")
    s = s.replace("```", " ")
    s = re.sub(r"[#*_>`\[\]\(\):;,]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def parse_generation_label(row: Dict, response: str) -> Optional[str]:
    text = response or ""
    upper = text.upper()
    if "ANSWER:" in upper:
        upper = upper.split("ANSWER:")[-1]
    upper = normalize_text(upper)[:120]

    if row.get("category") == "arithmetic":
        m = re.search(r"-?\d+", upper)
        return m.group(0) if m else None

    for label in row["valid_labels"]:
        alias_list = row.get("aliases", {}).get(label, [label])
        for alias in alias_list:
            alias_u = normalize_text(alias)
            if alias_u and re.search(rf"\b{re.escape(alias_u)}\b", upper):
                return label
    return None


def score_candidate_label(model, tokenizer, prompt_text: str, label: str, max_input_length: int) -> float:
    device = get_input_device(model)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_input_length, add_special_tokens=True)
    full_ids = tokenizer(prompt_text + " " + label, return_tensors="pt", truncation=True, max_length=max_input_length, add_special_tokens=True)

    prompt_input_ids = prompt_ids["input_ids"].to(device)
    full_input_ids = full_ids["input_ids"].to(device)
    full_attention = full_ids.get("attention_mask")
    if full_attention is not None:
        full_attention = full_attention.to(device)

    prompt_len = prompt_input_ids.shape[1]
    full_len = full_input_ids.shape[1]
    if full_len <= prompt_len:
        return float("-inf")

    with torch.no_grad():
        outputs = model(input_ids=full_input_ids, attention_mask=full_attention, use_cache=False)
        logits = outputs.logits[:, :-1, :].float()
        labels = full_input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        start = max(prompt_len - 1, 0)
        cont = token_log_probs[:, start:]
        if cont.numel() == 0:
            return float("-inf")
        return float(cont.mean().item())


def choose_label_by_score(model, tokenizer, row: Dict, prompt_text: str, max_input_length: int) -> Optional[str]:
    best_label = None
    best_score = float("-inf")
    for label in row["valid_labels"]:
        score = score_candidate_label(model, tokenizer, prompt_text, label, max_input_length=max_input_length)
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


def generate_text(model, tokenizer, prompt_text: str, max_input_length: int, max_new_tokens: int) -> str:
    device = get_input_device(model)
    batch = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
        add_special_tokens=True,
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        out = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id,
        )
    gen_ids = out[0, batch["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def evaluate_prompts(model, tokenizer, prompts: List[Dict], max_input_length: int, max_new_tokens: int, progress_every: int, use_chat_template: bool) -> Dict:
    details = []
    gen_valid = gen_correct = gen_nonempty = 0
    choice_correct = 0
    cat = {}

    for idx, row in enumerate(prompts, start=1):
        prompt_text = maybe_chat_wrap(tokenizer, row["prompt"], use_chat_template=use_chat_template)
        choice_pred = choose_label_by_score(model, tokenizer, row, prompt_text, max_input_length=max_input_length)
        choice_ok = choice_pred == row["gold"]
        choice_correct += int(choice_ok)

        response = generate_text(model, tokenizer, prompt_text, max_input_length=max_input_length, max_new_tokens=max_new_tokens)
        gen_pred = parse_generation_label(row, response)
        gen_is_nonempty = bool((response or "").strip())
        gen_is_valid = gen_pred is not None
        gen_is_correct = gen_pred == row["gold"]

        gen_nonempty += int(gen_is_nonempty)
        gen_valid += int(gen_is_valid)
        gen_correct += int(gen_is_correct)

        category = row.get("category", "unknown")
        if category not in cat:
            cat[category] = {"n": 0, "choice_correct": 0, "gen_valid": 0, "gen_correct": 0}
        cat[category]["n"] += 1
        cat[category]["choice_correct"] += int(choice_ok)
        cat[category]["gen_valid"] += int(gen_is_valid)
        cat[category]["gen_correct"] += int(gen_is_correct)

        details.append({
            "id": row["id"],
            "category": category,
            "gold": row["gold"],
            "choice_pred": choice_pred,
            "choice_correct": bool(choice_ok),
            "generation_pred": gen_pred,
            "generation_valid": bool(gen_is_valid),
            "generation_correct": bool(gen_is_correct),
            "response": response,
        })

        if progress_every > 0 and idx % progress_every == 0:
            print(f"  evaluated {idx}/{len(prompts)} prompts")

    per_category = {}
    for k, v in cat.items():
        n = v["n"]
        per_category[k] = {
            "n": n,
            "choice_accuracy": v["choice_correct"] / max(n, 1),
            "generation_valid_rate": v["gen_valid"] / max(n, 1),
            "generation_accuracy_over_all": v["gen_correct"] / max(n, 1),
            "generation_accuracy_valid_only": v["gen_correct"] / max(v["gen_valid"], 1),
        }

    return {
        "n": len(prompts),
        "choice_accuracy": choice_correct / max(len(prompts), 1),
        "generation_nonempty_rate": gen_nonempty / max(len(prompts), 1),
        "generation_valid_rate": gen_valid / max(len(prompts), 1),
        "generation_accuracy_over_all": gen_correct / max(len(prompts), 1),
        "generation_accuracy_valid_only": gen_correct / max(gen_valid, 1),
        "invalid_examples": int(len(prompts) - gen_valid),
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
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--max-prompts", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--raw-only", action="store_true")
    ap.add_argument("--enforce-quality-gate", action="store_true")
    ap.add_argument("--min-raw-choice-accuracy", type=float, default=0.60)
    ap.add_argument("--min-raw-generation-valid-rate", type=float, default=0.60)
    ap.add_argument("--use-chat-template", choices=["auto", "yes", "no"], default="auto")
    args = ap.parse_args()

    set_global_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    prompts = read_jsonl(args.prompts)
    if args.max_prompts > 0:
        prompts = prompts[:args.max_prompts]

    use_chat_template = False
    if args.use_chat_template == "yes":
        use_chat_template = True
    elif args.use_chat_template == "auto":
        use_chat_template = args.model_type in {"mistral", "tinyllama"}

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
        use_chat_template=use_chat_template,
    )

    quality_gate = {
        "min_raw_choice_accuracy": args.min_raw_choice_accuracy,
        "min_raw_generation_valid_rate": args.min_raw_generation_valid_rate,
        "passed": bool(
            raw_eval["choice_accuracy"] >= args.min_raw_choice_accuracy and
            raw_eval["generation_valid_rate"] >= args.min_raw_generation_valid_rate
        )
    }

    results = {
        "experiment_info": {
            "model": args.model,
            "model_type": args.model_type,
            "timestamp": datetime.now().isoformat(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "load_policy": args.load_policy,
            "num_prompts": len(prompts),
            "use_chat_template": use_chat_template,
        },
        "selected_layers": selected_layers,
        "raw_model": raw_eval,
        "quality_gate": quality_gate,
        "methods": {},
    }

    json_path = os.path.join(args.output, f"stage3_utility_{os.path.basename(args.model)}.json")
    if args.raw_only or (args.enforce_quality_gate and not quality_gate["passed"]):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {json_path}")
        if args.enforce_quality_gate and not quality_gate["passed"]:
            print("Quality gate failed. Aborting attacked conditions.")
        return

    clean_snapshot = helper_pg.snapshot_selected_layers(model, selected_layers)
    simulator = AttackSimulatorLarge(EVAL_TEXTS, max_length=128)

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
            use_chat_template=use_chat_template,
        )
        clean_verify = verify_method(wm, model, args.model, wm_key)
        wm_snapshot = helper_pg.snapshot_selected_layers(model, selected_layers)

        method_out = {
            "clean": {"utility": clean_eval, "verification": clean_verify},
            "conditions": {"clean": {"utility": clean_eval, "verification": clean_verify}},
        }

        for condition_name in condition_names:
            if condition_name == "clean":
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
            elif attack_name != "none":
                raise ValueError(f"Unsupported attack function: {attack_name}")

            cond_eval = evaluate_prompts(
                model, tokenizer, prompts,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
                progress_every=args.progress_every,
                use_chat_template=use_chat_template,
            )
            cond_verify = verify_method(wm, model, args.model, wm_key)
            method_out["conditions"][condition_name] = {"utility": cond_eval, "verification": cond_verify}
            cleanup()

        results["methods"][method] = method_out

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    rows = [{
        "method": "raw",
        "condition": "raw",
        "choice_accuracy": raw_eval["choice_accuracy"],
        "generation_valid_rate": raw_eval["generation_valid_rate"],
        "generation_accuracy_over_all": raw_eval["generation_accuracy_over_all"],
        "generation_accuracy_valid_only": raw_eval["generation_accuracy_valid_only"],
        "invalid_examples": raw_eval["invalid_examples"],
        "verification_confidence": None,
    }]
    for method, payload in results["methods"].items():
        for cond, cpay in payload["conditions"].items():
            rows.append({
                "method": method,
                "condition": cond,
                "choice_accuracy": cpay["utility"]["choice_accuracy"],
                "generation_valid_rate": cpay["utility"]["generation_valid_rate"],
                "generation_accuracy_over_all": cpay["utility"]["generation_accuracy_over_all"],
                "generation_accuracy_valid_only": cpay["utility"]["generation_accuracy_valid_only"],
                "invalid_examples": cpay["utility"]["invalid_examples"],
                "verification_confidence": cpay["verification"]["confidence"],
            })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output, "stage3_utility_summary.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5))
    for method in df["method"].unique():
        sub = df[df["method"] == method]
        plt.plot(sub["condition"], sub["choice_accuracy"], marker="o", label=method)
    plt.ylabel("Choice accuracy")
    plt.title("Stage 3 utility validation v2")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(args.output, "stage3_choice_accuracy.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
