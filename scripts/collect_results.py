"""Collect all weight quantization experiment results into summary tables.

Usage:
    python scripts/collect_results.py [--results-dir results/]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Collect MoQ experiment results")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Root results directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    all_results: dict[str, dict] = {}

    # Scan for eval_results.json files
    for eval_dir in sorted(results_dir.glob("*_eval")):
        results_file = eval_dir / "eval_results.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
            name = eval_dir.name.replace("_eval", "")
            all_results[name] = data

    if not all_results:
        print("No results found.")
        return

    # ── Table 1: All results ────────────────────────────────────────
    print("=" * 100)
    print("MoQ Experiment Results Summary")
    print("=" * 100)
    print(f"{'Experiment':<45s} {'PPL':>10s} {'HellaSwag':>10s} {'PIQA':>10s}")
    print("-" * 100)

    # FP baseline (from first result that has it)
    fp_ppl, fp_hella, fp_piqa = None, None, None
    for data in all_results.values():
        if "full_precision" in data:
            fp = data["full_precision"]
            fp_ppl = fp.get("ppl_wikitext2")
            zs = fp.get("zero_shot", {})
            fp_hella = zs.get("hellaswag")
            fp_piqa = zs.get("piqa")
            if fp_ppl:
                break

    if fp_ppl:
        print(f"{'FP32 (baseline)':<45s} {fp_ppl:>10.2f} {fp_hella:>10.4f} {fp_piqa:>10.4f}")
        print("-" * 100)

    # Group: existing activation-only results
    act_only = {k: v for k, v in all_results.items() if "_wt_" not in k and "combined" not in k}
    if act_only:
        print("\n  --- Activation-Only Results ---")
        for name, data in sorted(act_only.items()):
            q = data.get("quantized", {})
            ppl = q.get("ppl_wikitext2", float("nan"))
            zs = q.get("zero_shot", {})
            hella = zs.get("hellaswag", float("nan"))
            piqa = zs.get("piqa", float("nan"))
            print(f"  {name:<43s} {ppl:>10.2f} {hella:>10.4f} {piqa:>10.4f}")

    # Group: weight-only results
    wt_only = {k: v for k, v in all_results.items() if "_wt_" in k}
    if wt_only:
        print("\n  --- Weight-Only Results ---")
        for name, data in sorted(wt_only.items()):
            q = data.get("quantized", {})
            ppl = q.get("ppl_wikitext2", float("nan"))
            zs = q.get("zero_shot", {})
            hella = zs.get("hellaswag", float("nan"))
            piqa = zs.get("piqa", float("nan"))
            print(f"  {name:<43s} {ppl:>10.2f} {hella:>10.4f} {piqa:>10.4f}")

    # Group: combined results
    combined = {k: v for k, v in all_results.items() if "combined" in k}
    if combined:
        print("\n  --- Combined (Activation + Weight) Results ---")
        for name, data in sorted(combined.items()):
            q = data.get("quantized", {})
            ppl = q.get("ppl_wikitext2", float("nan"))
            zs = q.get("zero_shot", {})
            hella = zs.get("hellaswag", float("nan"))
            piqa = zs.get("piqa", float("nan"))
            print(f"  {name:<43s} {ppl:>10.2f} {hella:>10.4f} {piqa:>10.4f}")

    print("\n" + "=" * 100)

    # ── Save as JSON ────────────────────────────────────────────────
    summary_path = results_dir / "summary.json"
    summary = {}
    for name, data in sorted(all_results.items()):
        q = data.get("quantized", {})
        summary[name] = {
            "ppl": q.get("ppl_wikitext2"),
            "hellaswag": q.get("zero_shot", {}).get("hellaswag"),
            "piqa": q.get("zero_shot", {}).get("piqa"),
        }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
