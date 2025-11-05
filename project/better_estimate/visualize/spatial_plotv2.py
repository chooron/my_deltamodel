import json
from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import os
import sys
from dotenv import load_dotenv

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import cartopy.crs as ccrs

font_family = 'Times New Roman'
plt.rcParams.update({
    'font.family': font_family,
    'font.serif': [font_family],
    'mathtext.fontset': 'custom',
    'mathtext.rm': font_family,
    'mathtext.it': font_family,
    'mathtext.bf': font_family,
    'axes.unicode_minus': False,
})

load_dotenv()
sys.path.append(os.getenv("PROJ_PATH"))

from dmg.core.data.loaders import HydroLoader
from dmg.core.post.plot_line import plot_time_series
from dmg.core.data import txt_to_array, load_json
from dmg.core.post import geoplot_single_metric
from project.better_estimate import load_config
from dmg.core.post.plot_geo import fetch_geo_data
# ------------------------------------------#
# Choose the metric to plot. (See available metrics printed above, or in the metrics_agg.json file).
lstm_config = load_config(r'conf/config_dhbv_lstm.yaml')
lstm_config['mode'] = 'test'
lstm_metrics_path = os.path.join(lstm_config['out_path'], 'metrics.json')
lstm_metrics = load_json(lstm_metrics_path)
hope_config = load_config(r'conf/config_dhbv_hopev1.yaml')
hope_config['mode'] = 'test'
hope_metrics_path = os.path.join(hope_config['out_path'], 'metrics.json')
hope_metrics = load_json(hope_metrics_path)
loader = HydroLoader(lstm_config, test_split=True, overwrite=False)
loader.load_dataset()

def load_diff_data(geo_info_df, metric):
    diff_data = np.array(hope_metrics[metric]) - np.array(lstm_metrics[metric])
    diff_df = geo_info_df.copy()
    diff_df[metric] = diff_data
    return diff_df


geo_info_df = fetch_geo_data(lstm_config)
nse_diff = load_diff_data(geo_info_df, 'nse')
kge_diff = load_diff_data(geo_info_df, 'kge')
fhv_abs_diff = load_diff_data(geo_info_df, 'fhv_abs')

base_cmap1 = cm.get_cmap('RdBu')
colors1 = base_cmap1(np.linspace(0.05, 0.95, 256))  # 去掉最白部分
custom_cmap1 = mcolors.LinearSegmentedColormap.from_list("RdBu_no_white", colors1)

base_cmap2 = cm.get_cmap('RdBu_r')
colors2 = base_cmap1(np.linspace(0.05, 0.95, 256))  # 去掉最白部分
custom_cmap2 = mcolors.LinearSegmentedColormap.from_list("RdBu_r_no_white", colors2)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 9),
                         subplot_kw={'projection': ccrs.Mercator()},
                         constrained_layout=True)
# 5. Plot the evaluation results spatially.
geoplot_single_metric(nse_diff, 'nse', ax=axes[0],
                      vmin=-0.2, vmax=0.2, cmap=custom_cmap1, marker='^',
                      title="NSE Difference", cax_pos=[0.80, 0.07, 0.18, 0.03])
geoplot_single_metric(fhv_abs_diff, 'fhv_abs', ax=axes[1],
                      vmin=-20, vmax=20, cmap=base_cmap2, marker='^',
                      title="|FHV| Difference", cax_pos=[0.80, 0.07, 0.18, 0.03])
fig.savefig(os.path.join(os.getenv("PROJ_PATH"), "project/better_estimate/visualize/figures/spatial_diff_plot.png"),
            dpi=300)

# correlation
# basin_attr = loader.eval_dataset['c_nn_norm'].cpu().numpy()
# q_mean = np.mean(loader.eval_dataset['target'].cpu().numpy(), axis=0)
# basin_df = pd.DataFrame(data=basin_attr, columns=loader.nn_attributes)
# lstm_fhv = fetch_data(config=lstm_config, metric='fhv_abs')
# hopev1_fhv = fetch_data(config=hopev1_config, metric='fhv_abs')
# basin_df['q_mean'] = q_mean
# basin_df['fhv_abs1'] = np.log(lstm_fhv['fhv_abs'])
# basin_df['fhv_abs2'] = np.log(hopev1_fhv['fhv_abs'])
# plt.scatter(basin_df['q_mean'], basin_df['fhv_abs1'])
# plt.scatter(basin_df['q_mean'], basin_df['fhv_abs2'])
# plt.show()
# basin_df.corr()['fhv_abs1']
#

# nse_diff.loc[nse_diff['nse'] > 0.01]
# fhv_abs_diff.loc[kge_diff['kge'] > 0.01]

# nse_diff_values = nse_diff['nse'].values
# nse_diff_values = np.where(nse_diff_values > 0.5, 0.5, nse_diff_values)
# nse_diff_values = np.where(nse_diff_values < -0.5, -0.5, nse_diff_values)
# plt.hist(nse_diff_values, bins=100)
# plt.hist(fhv_abs_diff['fhv_abs'], bins=30)
# plt.show()
