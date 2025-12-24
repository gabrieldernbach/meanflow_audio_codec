import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from meanflow_audio_codec.configs.config import AnalysisConfig


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {path}")
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_float(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=np.float64)


def _aggregate(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        nfe = int(row["nfe"])
        improved = row["use_improved_mean_flow"].lower() == "true"
        key = (nfe, improved)
        grouped.setdefault(key, []).append(row)

    summary = []
    for (nfe, improved), items in sorted(grouped.items()):
        fid = _to_float(items, "fid")
        kid = _to_float(items, "kid")
        count = len(items)
        summary.append(
            {
                "nfe": nfe,
                "use_improved_mean_flow": improved,
                "count": count,
                "fid_mean": float(fid.mean()),
                "fid_ci95": float(1.96 * fid.std(ddof=1) / np.sqrt(count)) if count > 1 else 0.0,
                "kid_mean": float(kid.mean()),
                "kid_ci95": float(1.96 * kid.std(ddof=1) / np.sqrt(count)) if count > 1 else 0.0,
            }
        )
    return summary


def _write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric(rows: list[dict], metric: str, output_path: Path) -> None:
    for improved in [False, True]:
        subset = [r for r in rows if r["use_improved_mean_flow"] == improved]
        if not subset:
            continue
        nfes = np.array([r["nfe"] for r in subset], dtype=np.int32)
        means = np.array([r[f"{metric}_mean"] for r in subset], dtype=np.float64)
        cis = np.array([r[f"{metric}_ci95"] for r in subset], dtype=np.float64)
        label = "improved" if improved else "baseline"
        plt.errorbar(nfes, means, yerr=cis, marker="o", capsize=3, label=label)

    plt.xlabel("NFE")
    plt.ylabel(metric.upper())
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def analyze_results(config: AnalysisConfig) -> None:
    rows = _read_rows(config.metrics_csv)
    summary = _aggregate(rows)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary(config.output_dir / "summary.csv", summary)

    _plot_metric(summary, "fid", config.output_dir / "fid_vs_nfe.png")
    _plot_metric(summary, "kid", config.output_dir / "kid_vs_nfe.png")
    print(f"Wrote summary to {config.output_dir / 'summary.csv'}")


