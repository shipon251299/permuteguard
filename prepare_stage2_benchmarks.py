"""
Download/normalize benchmark datasets for offline reuse.

Usage examples:
  python prepare_stage2_benchmarks.py --tasks piqa,boolq,arc_easy,hellaswag --output benchmarks
  python prepare_stage2_benchmarks.py --tasks piqa --limit 500 --output benchmarks
"""
import argparse
import json
import os
from typing import Dict, Iterable, List

from datasets import load_dataset


def normalize_piqa(rows: Iterable[Dict]) -> List[Dict]:
    out = []
    for r in rows:
        out.append({
            "id": str(r.get("id", len(out))),
            "task": "piqa",
            "question": r["goal"],
            "choices": [r["sol1"], r["sol2"]],
            "answer_index": int(r["label"]),
        })
    return out


def normalize_boolq(rows: Iterable[Dict]) -> List[Dict]:
    out = []
    for r in rows:
        out.append({
            "id": str(r.get("id", len(out))),
            "task": "boolq",
            "context": r["passage"],
            "question": r["question"],
            "choices": ["no", "yes"],
            "answer_index": 1 if bool(r["answer"]) else 0,
        })
    return out


def normalize_arc(rows: Iterable[Dict]) -> List[Dict]:
    out = []
    label_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    for r in rows:
        texts = r["choices"]["text"]
        labels = r["choices"]["label"]
        answer = r["answerKey"]
        if answer not in label_to_idx:
            continue
        idx = labels.index(answer) if answer in labels else label_to_idx[answer]
        out.append({
            "id": str(r.get("id", len(out))),
            "task": "arc_easy",
            "question": r["question"],
            "choices": texts,
            "answer_index": int(idx),
        })
    return out


def normalize_hellaswag(rows: Iterable[Dict]) -> List[Dict]:
    out = []
    for r in rows:
        out.append({
            "id": str(r.get("ind", len(out))),
            "task": "hellaswag",
            "context": r["ctx"],
            "question": "Choose the most plausible ending.",
            "choices": list(r["endings"]),
            "answer_index": int(r["label"]),
        })
    return out


TASK_MAP = {
    "piqa": ("piqa", "validation", normalize_piqa),
    "boolq": ("google/boolq", "validation", normalize_boolq),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", normalize_arc),
    "hellaswag": ("hellaswag", "validation", normalize_hellaswag),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="piqa,boolq,arc_easy,hellaswag")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 means full split")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    for task in tasks:
        if task not in TASK_MAP:
            raise ValueError(f"Unsupported task: {task}")
        ds_name, split_name, normalizer = TASK_MAP[task]
        print(f"Loading {task} from {ds_name} split={split_name}")
        ds = load_dataset(ds_name, split=split_name)
        if args.limit > 0:
            ds = ds.select(range(min(args.limit, len(ds))))
        rows = normalizer(ds)
        out_path = os.path.join(args.output, f"{task}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
