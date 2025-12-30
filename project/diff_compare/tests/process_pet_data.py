import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm


def extract_forcing_global_search(
    id_file_path,
    base_data_path,
    output_file="camels_forcing_custom.pkl",
    start_date="1980-10-01",
    end_date="2014-09-30",
):
    """
    全目录扫描提取 CAMELS forcing 数据。
    不假设 ID 与文件夹的对应关系，而是先建立全局索引，确保找到文件。
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

    # --- 2. 建立全局文件索引 (Pre-indexing) ---
    # 目的：扫描所有 01-18 文件夹，建立 ID -> 文件路径 的映射
    print("Building global file index (Scanning all HUC folders)...")

    id_to_path_map = {}
    huc_dirs = [f"{i:02d}" for i in range(1, 19)]  # 01 到 18

    for huc in tqdm(huc_dirs, desc="Indexing Folders"):
        dir_path = os.path.join(base_data_path, huc)
        if not os.path.exists(dir_path):
            continue

        # 获取该文件夹下所有 model_output 文件
        try:
            files = os.listdir(dir_path)
            for f in files:
                if f.endswith("_05_model_output.txt"):
                    # 解析文件名获取 ID (例如: 01013500_05_model_output.txt -> 01013500)
                    bid = f.split("_")[0]
                    # 存入字典: Key=ID, Value=完整路径
                    id_to_path_map[bid] = os.path.join(dir_path, f)
        except Exception as e:
            print(f"Error scanning directory {dir_path}: {e}")

    print(f"Index built. Found {len(id_to_path_map)} unique files in total.")

    # --- 3. 准备时间轴与输出数组 ---
    target_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    num_steps = len(target_dates)
    final_data = np.full(
        (len(target_ids), num_steps, 3), np.nan, dtype=np.float32
    )

    missing_count = 0

    # --- 4. 提取数据 ---
    for idx, bid in enumerate(tqdm(target_ids, desc="Extracting Data")):
        # 4.1 直接从索引中查找路径 (替代之前的 HUC 推断)
        fpath = id_to_path_map.get(bid)

        if fpath is None:
            # 如果在所有文件夹里都没找到这个 ID
            missing_count += 1
            # print(f"Missing: {bid}")
            continue

        # 推断对应的 parameter 文件路径 (用于读取 PET 系数)
        # 假设 parameter 文件和 output 文件在同一个目录下
        param_path = fpath.replace(
            "_05_model_output.txt", "_05_model_parameters.txt"
        )

        try:
            # 4.2 读取 Forcing
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

            # 构建时间索引
            df["Date"] = pd.to_datetime(
                df[["YR", "MNTH", "DY"]].rename(
                    columns={"YR": "year", "MNTH": "month", "DY": "day"}
                )
            )
            df = df.set_index("Date")

            # 4.3 读取系数进行 PET 修正
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

            # 修正 PET
            if pet_coeff != 0:
                df["PET_adj"] = (1.26 / pet_coeff) * df["PET"]
            else:
                df["PET_adj"] = df["PET"]

            # 4.4 时间对齐
            df_aligned = df.reindex(target_dates)

            # 4.5 存入数组 [P, T, PET]
            data_block = df_aligned[["PRCP", "TAIR", "PET_adj"]].values.astype(
                np.float32
            )
            final_data[idx, :, :] = data_block

        except Exception as e:
            print(f"Error processing {bid} at {fpath}: {e}")

    # --- 5. 保存结果 ---
    save_dict = {
        "forcing": final_data,
        "variable_names": ["P", "T", "PET"],
        "basin_ids": target_ids,
        "dates": target_dates.values,
    }

    with open(output_file, "wb") as f:
        pickle.dump(save_dict, f)

    print("\nProcessing Complete.")
    print(f"Output Shape: {final_data.shape}")
    if missing_count > 0:
        print(
            f"Warning: {missing_count} Basin IDs were not found in any folder."
        )
    print(f"Saved to: {output_file}")


# --- 运行配置 ---
if __name__ == "__main__":
    # ID 文件路径
    ID_FILE = r"E:\PaperCode\dpl-project\generic_deltamodel\data\gage_id.npy"

    # CAMELS 根目录 (包含 01-18 文件夹的目录)
    BASE_DATA_DIR = r"G:\Dataset\CAMELS_US\model_output_daymet\model_output\flow_timeseries\daymet"

    # 输出文件
    OUTPUT_PKL = r"E:\PaperCode\dpl-project\generic_deltamodel\data\camels_forcing_v2.pkl"

    extract_forcing_global_search(
        id_file_path=ID_FILE,
        base_data_path=BASE_DATA_DIR,
        output_file=OUTPUT_PKL,
        start_date="1980-10-01",
        end_date="2014-09-30",
    )
