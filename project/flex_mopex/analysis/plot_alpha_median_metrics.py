import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_REFERENCE_JSON = Path(
    "/workspace/my_deltamodel/project/flex_mopex/output/flex_mopex_v1/alpha_0/"
    "camels_671/train1980-1995/no_multi/"
    "MultiHeadNet_E50_R365_B100_n4_noLn_noWU_42/FlexMopexV1/"
    "NseAicBatchLoss/stat/test1995-2010_Ep50/metrics_agg.json"
)

DEFAULT_ALPHAS = [
    "1",
    "0.5",
    "0.1",
    "0.07",
    "0.05",
    "0.03",
    "0.01",
    "0.007",
    "0.005",
    "0.003",
    "0.001",
    "0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect alpha-wise median r2/kge values and plot them."
    )
    parser.add_argument(
        "--reference-json",
        type=Path,
        default=DEFAULT_REFERENCE_JSON,
        help="Path to one metrics_agg.json under an alpha_* folder.",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        default=DEFAULT_ALPHAS,
        help="Alpha values in plotting order.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save the CSV and PNG outputs.",
    )
    return parser.parse_args()


def split_alpha_path(reference_json: Path) -> tuple[Path, Path, str]:
    parts = list(reference_json.parts)
    alpha_index = next(
        (idx for idx, part in enumerate(parts) if part.startswith("alpha_")), None
    )
    if alpha_index is None:
        raise ValueError(
            f"Cannot find an 'alpha_*' directory in path: {reference_json}"
        )

    alpha_dir_name = parts[alpha_index]
    prefix = Path(*parts[:alpha_index])
    suffix = Path(*parts[alpha_index + 1 :])
    return prefix, suffix, alpha_dir_name


def load_metric_values(json_path: Path) -> dict[str, float]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "r2": float(data["r2"]["median"]),
        "kge": float(data["kge"]["median"]),
    }


def collect_rows(reference_json: Path, alphas: list[str]) -> tuple[list[dict], list[Path]]:
    prefix, suffix, _ = split_alpha_path(reference_json)
    rows = []
    missing_files = []

    for alpha in alphas:
        json_path = prefix / f"alpha_{alpha}" / suffix
        if not json_path.exists():
            missing_files.append(json_path)
            rows.append({"alpha": alpha, "r2_median": np.nan, "kge_median": np.nan})
            continue

        metrics = load_metric_values(json_path)
        rows.append(
            {
                "alpha": alpha,
                "r2_median": metrics["r2"],
                "kge_median": metrics["kge"],
            }
        )

    return rows, missing_files


def save_csv(rows: list[dict], output_csv: Path) -> None:
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "r2_median", "kge_median"])
        writer.writeheader()
        writer.writerows(rows)


def plot_bars(rows: list[dict], output_png: Path) -> None:
    labels = [row["alpha"] for row in rows]
    r2_values = [row["r2_median"] for row in rows]
    kge_values = [row["kge_median"] for row in rows]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(15, 6))
    bars_r2 = ax.bar(
        x - width / 2,
        r2_values,
        width=width,
        label="R2 median",
        color="#4C72B0",
    )
    bars_kge = ax.bar(
        x + width / 2,
        kge_values,
        width=width,
        label="KGE median",
        color="#DD8452",
    )

    ax.set_title("Median R2 and KGE Across Alpha Settings")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Metric Value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(bottom=min(0.0, np.nanmin(r2_values + kge_values) - 0.05))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for bars in (bars_r2, bars_kge):
        for bar in bars:
            height = bar.get_height()
            if np.isnan(height):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    reference_json = args.reference_json.resolve()

    if args.output_dir is None:
        prefix, _, _ = split_alpha_path(reference_json)
        output_dir = prefix / "alpha_median_summary"
    else:
        output_dir = args.output_dir.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    rows, missing_files = collect_rows(reference_json, args.alphas)
    output_csv = output_dir / "alpha_median_metrics.csv"
    output_png = output_dir / "alpha_median_metrics.png"

    save_csv(rows, output_csv)
    plot_bars(rows, output_png)

    print(f"Saved CSV: {output_csv}")
    print(f"Saved plot: {output_png}")

    if missing_files:
        print("\nMissing files:")
        for path in missing_files:
            print(path)

    print("\nExtracted values:")
    for row in rows:
        print(
            f"alpha={row['alpha']:>5}  "
            f"r2_median={row['r2_median']:.6f}  "
            f"kge_median={row['kge_median']:.6f}"
        )


if __name__ == "__main__":
    main()
