"""Extract best-train metrics for alpine1/alpine2 and summarize statistics."""

import json
import os
import numpy as np

# Configuration
LOSS_NAME = "KgeInverseLoss"
METRIC_KEY = "inv_kge"
TRAIN_DIR = "train1989-1998"
TEST_DIR = "test1999-2009"
N_BASINS = 559
N_MEMBERS = 128
TOP_K = 1

# Default result path (relative to this script) with env override
RESULT_PATH = "/workspace/my_deltamodel/project/diff_compare/output/camels_559/train1989-1998/no_multi/Calibrate_E100_R365_B100_n128_noLn_noWU_42"


def _load_json_file(
    filepath: str, key: str, n_basins: int=N_BASINS, n_members: int=N_MEMBERS
) -> np.ndarray:
    if not os.path.exists(filepath):
        return np.full((n_basins, n_members), np.nan)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().lstrip("\ufeff").strip()
        data = json.loads(content)
        if isinstance(data, str):
            data = json.loads(data)

        arr = np.array(data.get(key, []), dtype=float)

        return arr.reshape(n_members, n_basins).T
    except Exception as e:
        return np.full((n_basins, n_members), np.nan)

def _extract_best(train_path: str, test_path: str):
	"""Pick test metrics by train-best member per basin."""
	train_data = _load_json_file(train_path, METRIC_KEY)
	test_data = _load_json_file(test_path, METRIC_KEY)
	best_idx = np.nanargmax(train_data, axis=1)
	train_best = train_data[np.arange(N_BASINS), best_idx]
	test_best = test_data[np.arange(N_BASINS), best_idx]
	return train_best, test_best


def summarize(values: np.ndarray):
	return float(np.nanmedian(values)), float(np.nanmean(values))


def topk_test_median(train_data: np.ndarray, test_data: np.ndarray, top_k: int = TOP_K) -> float:
	"""Median of test metrics at top-k train indices across all basins."""
	all_vals = []
	for i in range(train_data.shape[0]):
		train_row = train_data[i]
		test_row = test_data[i]
		valid_mask = ~np.isnan(train_row)
		valid_idx = np.nonzero(valid_mask)[0]
		if valid_idx.size == 0:
			continue
		# pick up to top-k train members
		vals = train_row[valid_idx]
		k = min(top_k, vals.size)
		if vals.size > k:
			top_rel = np.argpartition(vals, -k)[-k:]
			top_idx = valid_idx[top_rel]
		else:
			top_idx = valid_idx
		all_vals.extend(test_row[top_idx].tolist())
	return float(np.nanmedian(np.array(all_vals))) if all_vals else np.nan


def main():
	models = ["alpine1", "alpine2"]
	print(f"Scanning base path: {RESULT_PATH}")

	for model in models:
		train_path = os.path.join(
			RESULT_PATH, model, LOSS_NAME, "stat", f"{TRAIN_DIR}_Ep100", "metrics.json"
		)
		test_path = os.path.join(
			RESULT_PATH, model, LOSS_NAME, "stat", f"{TEST_DIR}_Ep100", "metrics.json"
		)

		train_best, test_best = _extract_best(train_path, test_path)
		if train_best is None:
			print(f"[Skip] {model}: train metrics missing or invalid.")
			continue

		train_med, train_mean = summarize(train_best)
		test_med, test_mean = summarize(test_best)
		topk_test_med = topk_test_median(
			_load_json_file(train_path, METRIC_KEY),
			_load_json_file(test_path, METRIC_KEY),
			top_k=TOP_K,
		)

		print(f"\nModel: {model}")
		print(f"Train  median/mean: {train_med:.4f} / {train_mean:.4f}")
		print(f"Test   median/mean: {test_med:.4f} / {test_mean:.4f}")
		print(f"Test median @ top-{TOP_K} train members: {topk_test_med:.4f}")


if __name__ == "__main__":
	main()
