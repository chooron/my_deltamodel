from pathlib import Path

import pandas as pd


def main() -> None:
        input_path = Path("/workspace/my_deltamodel/project/diff_compare/plots/csv/stats_calc_time.csv")
        output_path = input_path.with_name("stats_calc_time_processed.csv")

        df = pd.read_csv(input_path)

        df["T_epoch"] = df["T_step"] * df["k"]
        df["f_update"] = 60.0 / df["T_step"]
        df["total_train_time"] = df["t_epoch"] * 100
        df["total_updates"] = df["batch_size"] * df["n_iter_ep"]

        df = df.drop(columns=["R_eff", "k"])
        df = df.sort_values(by=["model", "batch_size", "pred_len"])

        df.to_csv(output_path, index=False)
        print(df.head())


if __name__ == "__main__":
        main()