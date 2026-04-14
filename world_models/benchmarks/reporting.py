from __future__ import annotations

from typing import Dict, Any
import csv
import json


def export_csv(results: Dict[str, Any], path: str):
    seeds = results.get("seeds", {})
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "mean", "std", "episode_returns"])
        for seed, data in seeds.items():
            writer.writerow(
                [
                    seed,
                    data.get("mean", ""),
                    data.get("std", ""),
                    json.dumps(data.get("episode_returns", [])),
                ]
            )


def export_markdown(results: Dict[str, Any], path: str):
    seeds = results.get("seeds", {})
    lines = []
    lines.append("| seed | mean | std | episode_returns |")
    lines.append("|---:|---:|---:|:---|")
    for seed, data in seeds.items():
        lines.append(
            f"| {seed} | {data.get('mean', ''):.3f} | {data.get('std', ''):.3f} | {data.get('episode_returns', [])} |"
        )

    agg = results.get("aggregate", {})
    lines.append("")
    lines.append("**Aggregate**")
    lines.append(f"Mean: {agg.get('mean', 0.0):.3f}  ")
    lines.append(f"Median: {agg.get('median', 0.0):.3f}  ")
    lines.append(f"IQM: {agg.get('iqm', 0.0):.3f}  ")

    with open(path, "w") as f:
        f.write("\n".join(lines))
