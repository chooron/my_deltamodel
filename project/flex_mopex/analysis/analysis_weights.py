import numpy as np
from pathlib import Path

base_data_path = Path(
    "/workspace/my_deltamodel/project/flex_mopex/output/camels_671/train1980-1995/no_multi/MultiHeadNet_E50_R365_B100_n16_noLn_noWU_42/FlexMopexV1/alpha_0_05/NseAicBatchLoss/stat"
)

data_range = "test1995-2010_Ep50"

w_int_data = np.load(base_data_path / data_range / "w_int.npy")[0, :, :]
w_phen_data = np.load(base_data_path / data_range / "w_phen.npy")[0, :, :]
w_snow_data = np.load(base_data_path / data_range / "w_snow.npy")[0, :, :]
w_sub_data = np.load(base_data_path / data_range / "w_sub.npy")[0, :, :]
