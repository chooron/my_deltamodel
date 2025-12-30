import pickle
import numpy as np
import pandas as pd
import os
import json


def check_nans_in_period(
    pkl_path, subset_id_path, gage_id_path, start_year=1989, end_year=2010
):
    print(f"Loading data from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    forcing = data["forcing"]  # Shape (671, T, 3)
    dates = pd.to_datetime(data["dates"])
    variable_names = data.get("variable_names", ["P", "T", "PET"])

    # --- 1. Load Subset IDs ---
    print(f"Loading subset IDs from {subset_id_path}...")
    with open(subset_id_path, "r") as f:
        subset_ids = json.load(f)
    subset_ids = [str(sid).zfill(8) for sid in subset_ids]

    # --- 2. Load Global Gage IDs to Map Subset Indices ---
    print(f"Loading global gage IDs from {gage_id_path}...")
    global_gage_ids = np.load(gage_id_path, allow_pickle=True)
    global_gage_ids = [str(int(float(gid))).zfill(8) for gid in global_gage_ids]

    # Map subset IDs to indices in the 671-basin dataset
    id_to_idx = {gid: i for i, gid in enumerate(global_gage_ids)}
    target_indices = []
    found_ids = []
    for sid in subset_ids:
        if sid in id_to_idx:
            target_indices.append(id_to_idx[sid])
            found_ids.append(sid)

    print(
        f"Subset size: {len(subset_ids)}, Found in global list: {len(target_indices)}"
    )

    # --- 3. Filter by Time Period ---
    time_mask = (dates.year >= start_year) & (dates.year <= end_year)
    period_forcing = forcing[target_indices, :, :][:, time_mask, :]
    period_dates = dates[time_mask]

    # --- 4. Check for NaNs ---
    nan_mask = np.isnan(period_forcing)
    nan_count = nan_mask.sum()

    print("\n" + "=" * 50)
    print(f"   NaN Analysis Report ({start_year}-{end_year})")
    print("=" * 50)
    print(f"Total elements scanned: {period_forcing.size}")
    print(f"Total NaNs found: {nan_count}")

    if nan_count > 0:
        # Check by variable
        for i, var in enumerate(variable_names):
            v_nan = nan_mask[:, :, i].sum()
            if v_nan > 0:
                print(f"  - Variable '{var}': {v_nan} NaNs")

        # Check by Basin and Date
        # nan_mask indexed as (basin_in_subset, time, variable)
        b_idx, t_idx, v_idx = np.where(nan_mask)
        unique_nan_dates = sorted(list(set(period_dates[t_idx])))
        unique_nan_basins = sorted(list(set([found_ids[i] for i in b_idx])))

        print(f"\nAffected Basins ({len(unique_nan_basins)}):")
        print(
            unique_nan_basins[:10], "..." if len(unique_nan_basins) > 10 else ""
        )

        print(f"\nAffected Dates ({len(unique_nan_dates)}):")
        if len(unique_nan_dates) > 20:
            print(
                f"First 10: {[d.strftime('%Y-%m-%d') for d in unique_nan_dates[:10]]}"
            )
            print(
                f"Last 10:  {[d.strftime('%Y-%m-%d') for d in unique_nan_dates[-10:]]}"
            )
        else:
            print([d.strftime("%Y-%m-%d") for d in unique_nan_dates])

        # Sampling some specific instances
        print("\nFirst 10 NaN instances (Basin, Date, Variable):")
        for i in range(min(10, len(b_idx))):
            basin_id = found_ids[b_idx[i]]
            date_str = period_dates[t_idx[i]].strftime("%Y-%m-%d")
            var_name = variable_names[v_idx[i]]
            print(f"  {basin_id} | {date_str} | {var_name}")
    else:
        print(
            f"Congratulations! No NaNs found for the 559 basins in {start_year}-{end_year}."
        )


if __name__ == "__main__":
    PKL_PATH = r"E:\PaperCode\dpl-project\generic_deltamodel\data\camels_forcing_v2.pkl"
    SUBSET_PATH = (
        r"E:\PaperCode\dpl-project\generic_deltamodel\data\559sub_id.txt"
    )
    GAGE_PATH = r"E:\PaperCode\dpl-project\generic_deltamodel\data\gage_id.npy"

    check_nans_in_period(PKL_PATH, SUBSET_PATH, GAGE_PATH)
