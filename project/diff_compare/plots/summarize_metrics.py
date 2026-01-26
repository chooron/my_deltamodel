from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

NPARAM_INFO = {
    "alpine1": 4,
    "alpine2": 6,
    "australia": 8,
    "collie1": 1,
    "collie2": 4,
    "collie3": 6,
    "flexb": 9,
    "flexi": 10,
    "flexis": 12,
    "gr4j": 4,
    "gsfb": 8,
    "hbv96": 15,
    "hillslope": 7,
    "hymod": 5,
    "ihacres": 6,
    "modhydrolog": 15,
    "mopex1": 5,
    "mopex2": 7,
    "mopex3": 8,
    "mopex4": 10,
    "mopex5": 5,
    "newzealand1": 6,
    "newzealand2": 8,
    "penman": 4,
    "plateau": 8,
    "simhyd": 7,
    "smar": 8,
    "susannah1": 6,
    "susannah2": 6,
    "tank": 12,
    "tcm": 6,
    "topmodel": 7,
    "us1": 5,
    "vic": 10,
    "wetland": 2,
    "xinanjiang": 12,
}

STATE_INFO = {
    "alpine1": 1,
    "alpine2": 2,
    "australia": 3,
    "collie1": 1,
    "collie2": 1,
    "collie3": 2,
    "flexb": 4,
    "flexi": 5,
    "flexis": 5,
    "gr4j": 2,
    "gsfb": 4,
    "hbv96": 5,
    "hillslope": 4,
    "hymod": 5,
    "ihacres": 2,
    "modhydrolog": 5,
    "mopex1": 3,
    "mopex2": 4,
    "mopex3": 3,
    "mopex4": 5,
    "mopex5": 5,
    "newzealand1": 3,
    "newzealand2": 2,
    "penman": 1,
    "plateau": 4,
    "simhyd": 3,
    "smar": 5,
    "susannah1": 5,
    "susannah2": 5,
    "tank": 6,
    "tcm": 1,
    "topmodel": 2,
    "us1": 4,
    "vic": 2,
    "wetland": 6,
    "xinanjiang": 4,
}

NUMBER_INFO = {
    "alpine1": 6,
    "alpine2": 12,
    "australia": 19,
    "collie1": 1,
    "collie2": 3,
    "collie3": 11,
    "flexb": 21,
    "flexi": 26,
    "flexis": 34,
    "gr4j": 7,
    "gsfb": 20,
    "hbv96": 37,
    "hillslope": 13,
    "hymod": 29,
    "ihacres": 5,
    "modhydrolog": 36,
    "mopex1": 24,
    "mopex2": 30,
    "mopex3": 31,
    "mopex4": 32,
    "mopex5": 35,
    "newzealand1": 4,
    "newzealand2": 16,
    "penman": 17,
    "plateau": 15,
    "simhyd": 18,
    "smar": 40,
    "susannah1": 9,
    "susannah2": 10,
    "tank": 27,
    "tcm": 25,
    "topmodel": 14,
    "us1": 8,
    "vic": 22,
    "wetland": 2,
    "xinanjiang": 28,
}

COLUMN_ALIASES = {
    "hbv": "hbv96",
}

def load_metric_table(path: Path, aliases: Dict[str, str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if aliases:
        df = df.rename(columns=aliases)
    return df


def compute_median_var(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    numeric_df = df.drop(columns=["basin_id"], errors="ignore")
    for col in numeric_df.columns:
        series = pd.to_numeric(numeric_df[col], errors="coerce")
        series = series.clip(lower=-5)
        metrics[col] = {
            "median": series.median(),
            "var": series.var(ddof=0),
        }
    return metrics


def ordered_models(ordering: Dict[str, int]) -> Iterable[Tuple[str, int]]:
    return sorted(ordering.items(), key=lambda kv: kv[1])


def get_stat(stats: Dict[str, Dict[str, float]], model: str, key: str) -> float:
    return stats.get(model, {}).get(key)


def format_stat(stats: Dict[str, Dict[str, float]], model: str) -> str:
    median = get_stat(stats, model, "median")
    var = get_stat(stats, model, "var")
    if pd.isna(median) or pd.isna(var):
        return ""
    return f"{median:.2f}±{var:.2f}"


def build_summary() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parent
    csv_dir = base_dir / "csv"

    train_kge = load_metric_table(csv_dir / "dif_train_kge.csv")
    test_kge = load_metric_table(csv_dir / "dif_test_kge.csv")
    train_invkge = load_metric_table(csv_dir / "dif_train_invkge.csv")
    test_invkge = load_metric_table(csv_dir / "dif_test_invkge.csv")

    marrmot_train_kge = load_metric_table(csv_dir / "marrmot_train_kge.csv")
    marrmot_test_kge = load_metric_table(csv_dir / "marrmot_test_kge.csv")
    marrmot_train_invkge = load_metric_table(
        csv_dir / "marrmot_train_invkge.csv", aliases=COLUMN_ALIASES
    )
    marrmot_test_invkge = load_metric_table(
        csv_dir / "marrmot_test_invkge.csv", aliases=COLUMN_ALIASES
    )

    train_kge_stats = compute_median_var(train_kge)
    test_kge_stats = compute_median_var(test_kge)
    train_invkge_stats = compute_median_var(train_invkge)
    test_invkge_stats = compute_median_var(test_invkge)

    marrmot_train_kge_stats = compute_median_var(marrmot_train_kge)
    marrmot_test_kge_stats = compute_median_var(marrmot_test_kge)
    marrmot_train_invkge_stats = compute_median_var(marrmot_train_invkge)
    marrmot_test_invkge_stats = compute_median_var(marrmot_test_invkge)

    rows = []
    for idx, (model, _) in enumerate(ordered_models(NUMBER_INFO), start=1):
        rows.append(
            {
                "idx": idx,
                "model": model,
                "n_params": NPARAM_INFO.get(model),
                "n_states": STATE_INFO.get(model),
                "kge_train": format_stat(train_kge_stats, model),
                "kge_test": format_stat(test_kge_stats, model),
                "kge_marrmot_train": format_stat(marrmot_train_kge_stats, model),
                "kge_marrmot_test": format_stat(marrmot_test_kge_stats, model),
                "invkge_train": format_stat(train_invkge_stats, model),
                "invkge_test": format_stat(test_invkge_stats, model),
                "invkge_marrmot_train": format_stat(marrmot_train_invkge_stats, model),
                "invkge_marrmot_test": format_stat(marrmot_test_invkge_stats, model),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    summary_df = build_summary()
    base_dir = Path(__file__).resolve().parent
    output_path = base_dir / "csv" / "metrics_summary.csv"
    summary_df.to_csv(output_path, index=False)
    print(summary_df.head())
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
