import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd


def create_summary_and_plots(results: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    rows = []
    for attack, payload in results["attack_results"].items():
        rows.append(
            {
                "attack": attack,
                "confidence_mean": payload["confidence"]["mean"],
                "confidence_std": payload["confidence"]["std"],
                "perplexity_mean": payload["perplexity"]["mean"],
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
    plt.bar(df["attack"], df["utility_ok_rate"], alpha=0.55, label="Utility OK")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Rate")
    plt.ylim(0, 1.05)
    plt.title("Robustness rates by attack")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "robustness_rates.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(df["confidence_mean"], df["perplexity_mean"])
    for _, row in df.iterrows():
        plt.annotate(row["attack"], (row["confidence_mean"], row["perplexity_mean"]), fontsize=8)
    plt.xlabel("Mean confidence")
    plt.ylabel("Mean perplexity")
    plt.title("Confidence vs utility")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "confidence_vs_utility.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.results_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    create_summary_and_plots(results, args.output)


if __name__ == "__main__":
    main()
