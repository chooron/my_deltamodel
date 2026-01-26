import pandas as pd

train_path = "/workspace/my_deltamodel/project/diff_compare/plots/csv/dif_train_kge.csv"
test_path = "/workspace/my_deltamodel/project/diff_compare/plots/csv/dif_test_kge.csv"
model_name = "hbv96"
threshold = 0.5  # condition: train - test > threshold

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

if model_name not in train_df.columns or model_name not in test_df.columns:
	raise KeyError(f"Column {model_name} not found in both CSVs")

if "basin_id" in train_df.columns and "basin_id" in test_df.columns:
	merged = train_df[["basin_id", model_name]].merge(
		test_df[["basin_id", model_name]], on="basin_id", suffixes=("_train", "_test")
	)
	train_vals = merged[f"{model_name}_train"]
	test_vals = merged[f"{model_name}_test"]
else:
	# fallback: align by index
	train_vals = train_df[model_name]
	test_vals = test_df[model_name]

diff = train_vals - test_vals
mask = diff > threshold
count = int(mask.sum())
print(f"Rows with (train - test) {model_name} > {threshold}: {count}")