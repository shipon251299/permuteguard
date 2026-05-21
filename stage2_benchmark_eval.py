
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import torch


LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_row(row: Dict) -> Dict:
    if "question" in row and "choices" in row and "answer_index" in row:
        return row

    if "goal" in row and "sol1" in row and "sol2" in row and "label" in row:
        return {
            "id": str(row.get("id", "")),
            "task": "piqa",
            "question": row["goal"],
            "choices": [row["sol1"], row["sol2"]],
            "answer_index": int(row["label"]),
        }

    if "passage" in row and "question" in row and "answer" in row:
        return {
            "id": str(row.get("id", "")),
            "task": "boolq",
            "context": row["passage"],
            "question": row["question"],
            "choices": ["no", "yes"],
            "answer_index": 1 if bool(row["answer"]) else 0,
        }

    if "question" in row and "choices" in row and "answerKey" in row:
        choices = row["choices"]
        if isinstance(choices, dict) and "text" in choices and "label" in choices:
            texts = choices["text"]
            labels = choices["label"]
            answer = row["answerKey"]
            if answer in labels:
                idx = labels.index(answer)
                return {
                    "id": str(row.get("id", "")),
                    "task": "arc_easy",
                    "question": row["question"],
                    "choices": texts,
                    "answer_index": int(idx),
                }

    if "ctx" in row and "endings" in row and "label" in row:
        return {
            "id": str(row.get("ind", "")),
            "task": "hellaswag",
            "context": row["ctx"],
            "question": "Choose the most plausible ending.",
            "choices": list(row["endings"]),
            "answer_index": int(row["label"]),
        }

    raise ValueError(f"Unsupported benchmark row schema. Keys: {sorted(row.keys())}")


def get_input_device(model) -> torch.device:
    try:
        emb = model.get_input_embeddings()
        return emb.weight.device
    except Exception:
        return next(model.parameters()).device


def format_generation_prompt(row: Dict) -> Tuple[str, int]:
    row = normalize_row(row)
    context = row.get("context", "")
    question = row["question"]
    choices = row["choices"]
    answer_index = int(row["answer_index"])

    parts = [
        "Answer the multiple-choice question by replying with only the option letter.",
    ]
    if context:
        parts.append(f"Context: {context}")
    parts.append(f"Question: {question}")
    parts.append("Options:")
    for i, c in enumerate(choices):
        parts.append(f"{LABELS[i]}. {c}")
    parts.append("Answer:")
    return "\n".join(parts), answer_index


def parse_generated_label(text: str, num_choices: int) -> Optional[int]:
    if text is None:
        return None

    cleaned = text.strip()
    if not cleaned:
        return None

    # Best case: standalone letter
    m = re.search(r"\b([A-Z])\b", cleaned.upper())
    if m:
        letter = m.group(1)
        idx = LABELS.find(letter)
        return idx if 0 <= idx < num_choices else None

    # Fallback: first alphabetic character
    for ch in cleaned.upper():
        if "A" <= ch <= "Z":
            idx = LABELS.find(ch)
            return idx if 0 <= idx < num_choices else None

    return None


def generate_label(
    model,
    tokenizer,
    prompt: str,
    num_choices: int,
    max_input_length: int = 384,
    max_new_tokens: int = 3,
) -> Dict:
    device = get_input_device(model)
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_ids = out[0, input_ids.shape[1]:]
    gen_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    pred = parse_generated_label(gen_text, num_choices=num_choices)
    return {
        "generated_text": gen_text,
        "pred_index": pred,
        "generated_token_ids": new_ids.detach().cpu().tolist(),
    }


def _label_token_debug(tokenizer, num_choices: int) -> Dict[str, List[int]]:
    return {
        f" {LABELS[i]}": tokenizer.encode(f" {LABELS[i]}", add_special_tokens=False)
        for i in range(num_choices)
    }


def select_stable_subset_generation(
    model,
    tokenizer,
    task_name: str,
    rows: List[Dict],
    target_n: int,
    max_scan: int = 300,
    max_input_length: int = 384,
    max_new_tokens: int = 3,
    progress_every: int = 10,
) -> Dict:
    selected_rows: List[Dict] = []
    debug: List[Dict] = []

    scan_n = min(len(rows), max_scan)
    for i in range(scan_n):
        if progress_every > 0 and (i + 1) % progress_every == 0:
            print(f"  subset scan {i + 1}/{scan_n}; selected={len(selected_rows)}")

        row = normalize_row(rows[i])
        prompt, _ = format_generation_prompt(row)
        gen = generate_label(
            model,
            tokenizer,
            prompt,
            num_choices=len(row["choices"]),
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
        )
        valid = gen["pred_index"] is not None

        if len(debug) < 3:
            debug.append(
                {
                    "id": row.get("id", ""),
                    "task": row.get("task", task_name),
                    "label_token_ids": _label_token_debug(tokenizer, len(row["choices"])),
                    "generated_text": gen["generated_text"],
                    "generated_token_ids": gen["generated_token_ids"],
                    "pred_index": gen["pred_index"],
                    "valid": bool(valid),
                }
            )

        if valid:
            selected_rows.append(row)

        if len(selected_rows) >= target_n:
            break

    return {
        "selected_rows": selected_rows[:target_n],
        "meta": {
            "selected_n": len(selected_rows[:target_n]),
            "target_n": int(target_n),
            "scanned_n": int(min(scan_n, i + 1 if scan_n > 0 else 0)),
            "examples_debug": debug,
        },
    }


def evaluate_fixed_subset_generation(
    model,
    tokenizer,
    task_name: str,
    rows: Iterable[Dict],
    max_input_length: int = 384,
    max_new_tokens: int = 3,
    progress_every: int = 10,
) -> Dict:
    rows = list(rows)
    details = []
    correct = 0
    valid_count = 0
    invalid_count = 0

    for i, raw_row in enumerate(rows):
        if progress_every > 0 and i > 0 and i % progress_every == 0:
            print(f"  evaluated {i}/{len(rows)} fixed examples")

        row = normalize_row(raw_row)
        prompt, gold = format_generation_prompt(row)
        gen = generate_label(
            model,
            tokenizer,
            prompt,
            num_choices=len(row["choices"]),
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
        )
        pred = gen["pred_index"]
        valid = pred is not None
        if valid:
            valid_count += 1
            ok = pred == gold
            correct += int(ok)
        else:
            invalid_count += 1
            ok = False

        details.append(
            {
                "id": row.get("id", str(i)),
                "task": row.get("task", task_name),
                "gold": int(gold),
                "pred": pred,
                "correct": bool(ok),
                "valid": bool(valid),
                "generated_text": gen["generated_text"],
                "generated_token_ids": gen["generated_token_ids"],
                "error": None if valid else "invalid_generated_label",
            }
        )

    accuracy_over_subset = correct / max(len(rows), 1)
    accuracy_valid_only = correct / max(valid_count, 1)

    return {
        "n": len(rows),
        "evaluated_fixed_subset": len(rows),
        "valid_examples": int(valid_count),
        "invalid_examples": int(invalid_count),
        "accuracy_over_subset": float(accuracy_over_subset),
        "accuracy_valid_only": float(accuracy_valid_only),
        "details": details,
    }


def evaluate_task_files_generation(
    model,
    tokenizer,
    task_to_rows: Dict[str, List[Dict]],
    max_input_length: int = 384,
    max_new_tokens: int = 3,
    progress_every: int = 10,
) -> Dict:
    results = {}
    acc_subset = []
    acc_valid = []

    for task_name, rows in task_to_rows.items():
        print(f"\nEvaluating fixed subset for task: {task_name}")
        out = evaluate_fixed_subset_generation(
            model,
            tokenizer,
            task_name,
            rows,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            progress_every=progress_every,
        )
        results[task_name] = out
        acc_subset.append(out["accuracy_over_subset"])
        acc_valid.append(out["accuracy_valid_only"])

    macro_subset = sum(acc_subset) / max(len(acc_subset), 1)
    macro_valid = sum(acc_valid) / max(len(acc_valid), 1)

    return {
        "tasks": results,
        "macro_accuracy_over_subset": float(macro_subset),
        "macro_accuracy_valid_only": float(macro_valid),
        "num_tasks": len(task_to_rows),
    }
