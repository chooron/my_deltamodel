import numpy as np
import matplotlib.pyplot as plt

train_data_path = "/workspace/my_deltamodel/project/diff_compare/output/camels_559/train1989-1998/no_multi/Calibrate_E100_R365_B100_n128_noLn_noWU_42/alpine1/KgeInverseLoss/stat/test1989-1998_Ep100/model_outputs.npz"
test_data_path = "/workspace/my_deltamodel/project/diff_compare/output/camels_559/train1989-1998/no_multi/Calibrate_E100_R365_B100_n128_noLn_noWU_42/alpine1/KgeInverseLoss/stat/test1999-2009_Ep100/model_outputs.npz"

T_LEN = 3653
N_MEM = 128
N_BASIN = 559


def load_streamflow(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    stream = data["streamflow"]
    print(stream.shape)
    return stream.reshape(-1, N_BASIN, N_MEM)


def plot_one_member(
    train_stream: np.ndarray,
    test_stream: np.ndarray,
    member_idx: int = 0,
    basin_idx: int = 0,
):
    train_ts = train_stream[:, member_idx, basin_idx]
    test_ts = test_stream[:, member_idx, basin_idx]
    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(train_ts.shape[0]), train_ts, label="train", color="tab:blue")
    plt.plot(
        np.arange(train_ts.shape[0], train_ts.shape[0] + test_ts.shape[0]),
        test_ts,
        label="test",
        color="tab:orange",
    )
    plt.xlabel("Time index")
    plt.ylabel("Streamflow")
    plt.title(f"Streamflow basin {basin_idx}, member {member_idx}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("streamflow_basin{}_member{}.png".format(basin_idx, member_idx))
    plt.close()


def main():
    train_stream = load_streamflow(train_data_path)
    test_stream = load_streamflow(test_data_path)
    plot_one_member(train_stream, test_stream, member_idx=0, basin_idx=0)


if __name__ == "__main__":
    main()
