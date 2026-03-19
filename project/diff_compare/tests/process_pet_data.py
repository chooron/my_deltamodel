import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm


def extract_forcing_global_search(
    id_file_path,
    base_data_path,
    old_data_path,
    output_file="camels_forcing_custom.pkl",
    start_date="1980-10-01",
    end_date="2014-09-30",
):
    """
    全目录扫描提取 CAMELS forcing 数据。
    如果新提取的数据存在缺失，使用旧数据进行填充。
    """

    # --- 1. 加载目标 ID ---
    print(f"Loading Target IDs from: {id_file_path}")
    try:
        raw_ids = np.load(id_file_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading npy file: {e}")
        return

    # 转换为 8 位字符串
    target_ids = [str(bid).strip().zfill(8) for bid in raw_ids]
    print(f"Target Basins to find: {len(target_ids)}")

    # --- 2. 加载旧数据作为背景填充 (Fallback) ---
    print(f"Loading old dataset for fallback: {old_data_path}")
    try:
        with open(old_data_path, "rb") as f:
            forcing_old, _, _ = pickle.load(f)
        print(f"Old data loaded. Shape: {forcing_old.shape}")
    except Exception as e:
        print(f"Error loading old dataset: {e}. Will initialize with NaNs.")
        forcing_old = None

    # --- 3. 建立全局文件索引 (Pre-indexing) ---
    print("Building global file index (Scanning all HUC folders)...")
    id_to_path_map = {}
    huc_dirs = [f"{i:02d}" for i in range(1, 19)]

    for huc in tqdm(huc_dirs, desc="Indexing Folders"):
        dir_path = os.path.join(base_data_path, huc)
        if not os.path.exists(dir_path):
            continue
        try:
            files = os.listdir(dir_path)
            for f in files:
                if f.endswith("_05_model_output.txt"):
                    bid = f.split("_")[0]
                    id_to_path_map[bid] = os.path.join(dir_path, f)
        except Exception as e:
            print(f"Error scanning directory {dir_path}: {e}")

    # --- 4. 准备时间轴与输出数组 ---
    target_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    num_steps = len(target_dates)

    # 使用旧数据初始化
    if forcing_old is not None:
        # 对齐旧数据到输出数组
        final_data = np.full(
            (len(target_ids), num_steps, 3), np.nan, dtype=np.float32
        )
        fill_len = min(forcing_old.shape[1], num_steps)
        final_data[:, :fill_len, :] = forcing_old[:, :fill_len, :].astype(
            np.float32
        )
        print(f"Initialized with old data (filled {fill_len} steps).")
    else:
        final_data = np.full(
            (len(target_ids), num_steps, 3), np.nan, dtype=np.float32
        )

    missing_count = 0
    updated_count = 0

    # --- 5. 提取数据并覆盖 ---
    for idx, bid in enumerate(
        tqdm(target_ids, desc="Extracting & Updating Data")
    ):
        fpath = id_to_path_map.get(bid)
        if fpath is None:
            missing_count += 1
            continue

        param_path = fpath.replace(
            "_05_model_output.txt", "_05_model_parameters.txt"
        )
        try:
            df = pd.read_csv(
                fpath,
                sep="\s+",
                header=None,
                skiprows=1,
                names=[
                    "YR",
                    "MNTH",
                    "DY",
                    "HR",
                    "SWE",
                    "PRCP",
                    "RAIM",
                    "TAIR",
                    "PET",
                    "ET",
                    "MOD_RUN",
                    "OBS_RUN",
                ],
            )
            df["Date"] = pd.to_datetime(
                df[["YR", "MNTH", "DY"]].rename(
                    columns={"YR": "year", "MNTH": "month", "DY": "day"}
                )
            )
            df = df.set_index("Date")

            # PET 修正
            pet_coeff = 1.0
            if os.path.exists(param_path):
                try:
                    df_param = pd.read_csv(
                        param_path,
                        sep="\s+",
                        header=None,
                        names=["Param", "Value"],
                    )
                    if len(df_param) > 40:
                        pet_coeff = df_param.iloc[40]["Value"]
                except:
                    pass

            if pet_coeff != 0:
                df["PET_adj"] = (1.26 / pet_coeff) * df["PET"]
            else:
                df["PET_adj"] = df["PET"]

            # 时间对齐
            df_aligned = df.reindex(target_dates)
            data_block = df_aligned[["PRCP", "TAIR", "PET_adj"]].values.astype(
                np.float32
            )

            # --- 关键：仅在提取到非 NaN 数据时覆盖 ---
            # 找到有效数据的掩码
            valid_mask = ~np.isnan(data_block).any(axis=1)
            if valid_mask.any():
                final_data[idx, valid_mask, :] = data_block[valid_mask, :]
                updated_count += 1

        except Exception as e:
            print(f"Error processing {bid}: {e}")

    # --- 6. 保存结果 ---
    save_dict = {
        "forcing": final_data,
        "variable_names": ["P", "T", "PET"],
        "basin_ids": target_ids,
        "dates": target_dates.values,
    }

    with open(output_file, "wb") as f:
        pickle.dump(save_dict, f)

    print(
        f"\nProcessing Complete. Updated {updated_count} basins with new data."
    )
    print(f"Output Shape: {final_data.shape}")
    if missing_count > 0:
        print(
            f"Warning: {missing_count} Basin IDs used old data ONLY (new files missing)."
        )
    print(f"Saved to: {output_file}")


# --- 运行配置 ---
if __name__ == "__main__":
    ID_FILE = r"E:\PaperCode\dpl-project\generic_deltamodel\data\gage_id.npy"
    BASE_DATA_DIR = r"G:\Dataset\CAMELS_US\model_output_daymet\model_output\flow_timeseries\daymet"
    OLD_DATA_FILE = (
        r"E:\PaperCode\dpl-project\generic_deltamodel\data\camels_dataset"
    )
    OUTPUT_PKL = r"E:\PaperCode\dpl-project\generic_deltamodel\data\camels_forcing_v2.pkl"

    extract_forcing_global_search(
        id_file_path=ID_FILE,
        base_data_path=BASE_DATA_DIR,
        old_data_path=OLD_DATA_FILE,
        output_file=OUTPUT_PKL,
        start_date="1980-10-01",
        end_date="2014-09-30",
    )
