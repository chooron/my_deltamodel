import numpy as np

def max_diff(arr1: np.ndarray, arr2: np.ndarray) -> float:
    assert arr1.shape == arr2.shape, "Input arrays must have the same shape"
    return np.max(np.abs(arr1 - arr2))


# example
basin_num, time = 5, 100
a = np.load("/workspace/my_deltamodel/project/triton_accelerate/output/camels_671/train1980-1995/no_multi/AnnModel_E60_R365_B100_n16_noLn_noWU_42/HbvTriton/NseBatchLoss/stat/test1995-2010_Ep60/Qsimmu.npy")
b = np.load("/workspace/my_deltamodel/project/hydro_selection/output/camels_671/train1980-1995/no_multi/AnnModel_E50_R365_B100_n16_noLn_noWU_42/Hbv/NseBatchLoss/stat/test1995-2010_Ep50/Qsimmu.npy")

print("Max difference:", max_diff(a, b))
