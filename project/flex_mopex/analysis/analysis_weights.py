import numpy as np
from pathlib import Path

base_data_path = Path(
    "/workspace/my_deltamodel/project/flex_mopex/output/camels_671/train1980-1995/no_multi/MultiHeadNetDyn_E50_R365_B100_n16_noLn_noWU_42/FlexMopexV2/NseDynAicBatchLoss/stat"
)

data_range = "test1995-2010_Ep50"

w_int_data = np.load(base_data_path / data_range / "w_int.npy")

print(w_int_data.shape)


