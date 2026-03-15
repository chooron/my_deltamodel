"""调试 MC Dropout metrics 保存问题"""
import numpy as np
import os

mc_dropout_dir = "/workspace/my_deltamodel/project/deal_hydro/outputv2/camels_559/train1989-1998/no_multi/Parameterize_E100_R365_B100_n1_noLn_noWU_42/hbv96/KgeBatchLoss/stat/test1999-2009_Ep100/mc_dropout"

# 检查文件
for dataset in ["train", "eval"]:
    metrics_file = os.path.join(mc_dropout_dir, f"{dataset}_metrics_samples.npz")

    if os.path.exists(metrics_file):
        print(f"\n{dataset.upper()} metrics file:")
        print(f"  Path: {metrics_file}")
        print(f"  Size: {os.path.getsize(metrics_file) / 1024:.2f} KB")

        data = np.load(metrics_file)
        print(f"  Keys: {list(data.keys())}")

        for key in data.keys():
            print(f"    {key}: shape={data[key].shape}, dtype={data[key].dtype}")
    else:
        print(f"\n{dataset.upper()} metrics file not found!")

# 检查 summary 文件
for dataset in ["train", "eval"]:
    summary_file = os.path.join(mc_dropout_dir, f"{dataset}_metrics_summary.txt")
    if os.path.exists(summary_file):
        print(f"\n{dataset.upper()} summary:")
        with open(summary_file, 'r') as f:
            print(f.read())
