"""
展示LSTM/S4D模型利用事后解释方法的结果图
"""

import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
from pathlib import Path

# attr shape = (730, 730, 48, 38)
# choose one output point:
result_path = (
    Path(__file__).parent.parent / "interpret" / "results" / "lstm" / "1031500"
)
attr = np.load(result_path / "tint_contribs.npy")
# attr = np.load(result_path / "contributions.npy")
attr = np.median(attr.reshape(730, 730, 38, 3, 16), axis=-1)
heat = attr[730 - 100, :, :, 1]
