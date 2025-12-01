import json
import logging
import os
import pickle
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from numpy.typing import NDArray
from sklearn.exceptions import DataDimensionalityWarning

load_dotenv()
from dmg.core.data.data import intersect
from dmg.core.data.loaders.base import BaseLoader

log = logging.getLogger(__name__)


class DlLoader(BaseLoader):
    """Data loader for pure deep learning models.
    
    This loader is designed for pure neural network prediction tasks,
    loading only the data needed for sequence-to-sequence or 
    sequence-to-one predictions without physics model components.

    Parameters
    ----------
    config
        Configuration dictionary.
    test_split
        Whether to split data into training and testing sets.
    overwrite
        Whether to overwrite existing normalization statistics.

    NOTE: Supports CAMELS-style datasets.
    """
    def __init__(
        self,
        config: dict[str, Any],
        test_split: Optional[bool] = False,
        overwrite: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.test_split = test_split
        self.overwrite = overwrite
        
        self.supported_data = ['camels_671', 'camels_531', 'prism_671', 'prism_531']
        self.data_name = config['observations']['name']
        
        # Neural network model configurations
        # 从 delta_model.nn_model 读取配置
        self.nn_attributes = config['delta_model']['nn_model'].get('attributes', [])
        self.nn_forcings = config['delta_model']['nn_model'].get('forcings', [])
        self.forcing_names = self.config['observations']['all_forcings']
        self.attribute_names = self.config['observations']['all_attributes']

        self.target = config['train']['target']
        # use_log_norm 在 phy_model 下面
        self.log_norm_vars = config['delta_model']['phy_model'].get('use_log_norm', [])
        self.device = config['device']
        self.dtype = config['dtype']

        self.train_dataset = None
        self.eval_dataset = None
        self.dataset = None
        self.norm_stats = None

        if self.data_name not in self.supported_data:
            raise ValueError(f"Data source '{self.data_name}' not supported.")

        self.load_dataset()

    def load_dataset(self) -> None:
        """Load data into dictionary of nn input tensors."""
        mode = self.config['mode']
        
        if mode == 'simulation':
            self.dataset = self._preprocess_data(scope='simulation')
        elif self.test_split:
            self.train_dataset = self._preprocess_data(scope='train')
            self.eval_dataset = self._preprocess_data(scope='test')
        elif mode in ['train', 'test']:
            self.train_dataset = self._preprocess_data(scope=mode)
        else:
            self.dataset = self._preprocess_data(scope='all')

    def _preprocess_data(
        self,
        scope: Optional[str],
    ) -> dict[str, torch.Tensor]:
        """Read data, preprocess, and return as tensors for models.
        
        Parameters
        ----------
        scope
            Scope of data to read, affects what timespan of data is loaded.
            
        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of data tensors for running models.
        """
        x_nn, c_nn, target = self.read_data(scope)

        # Normalize nn input data
        self.load_norm_stats(x_nn, c_nn, target)
        x_nn_norm, xc_nn_norm, c_nn_norm = self.normalize(x_nn, c_nn)

        # Build data dict of Torch tensors
        dataset = {
            'x_nn': self.to_tensor(x_nn),
            'c_nn': self.to_tensor(c_nn),
            'x_nn_norm': self.to_tensor(x_nn_norm),
            'xc_nn_norm': self.to_tensor(xc_nn_norm),
            'c_nn_norm': self.to_tensor(c_nn_norm),
            'target': self.to_tensor(target),
        }
        return dataset

    def read_data(
        self,
        scope: Optional[str],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Read data from the data file.
        
        Parameters
        ----------
        scope
            Scope of data to read, affects what timespan of data is loaded.

        Returns
        -------
        tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]
            Tuple of (x_nn, c_nn, target) data arrays.
        """
        try:
            data_path = os.path.join(os.getenv("DATA_PATH"), "camels_dataset")

            if scope == 'train':
                time = self.config['train_time']
            elif scope == 'test':
                time = self.config['test_time']
            elif scope == 'simulation':
                time = self.config['sim_time']
            elif scope == 'all':
                time = self.config['all_time']
            else:
                raise ValueError(
                    "Scope must be 'train', 'test', 'simulation', or 'all'."
                )
        except KeyError as e:
            raise ValueError(f"Key {e} for data path not in dataset config.") from e

        # Get time indices
        all_time = pd.date_range(
            self.config['all_time'][0],
            self.config['all_time'][-1],
            freq='d',
        )
        idx_start = all_time.get_loc(time[0])
        idx_end = all_time.get_loc(time[-1]) + 1

        # Load data
        with open(data_path, 'rb') as f:
            forcings, target, attributes = pickle.load(f)

        forcings = np.transpose(forcings[:, idx_start:idx_end], (1, 0, 2))

        # Forcings subset for nn model
        nn_forc_idx = []
        for forc in self.nn_forcings:
            if forc not in self.forcing_names:
                raise ValueError(
                    f"Forcing {forc} not in the list of all forcings."
                )
            nn_forc_idx.append(self.forcing_names.index(forc))

        # Attribute subset for nn model
        nn_attr_idx = []
        for attr in self.nn_attributes:
            if attr not in self.attribute_names:
                raise ValueError(
                    f"Attribute {attr} not in the list of all attributes."
                )
            nn_attr_idx.append(self.attribute_names.index(attr))

        x_nn = forcings[:, :, nn_forc_idx]
        c_nn = attributes[:, nn_attr_idx]
        target = np.transpose(target[:, idx_start:idx_end], (1, 0, 2))
        
        gage_info = np.load(os.path.join(os.getenv("DATA_PATH"), "gage_id.npy"))
        
        # Subset basins if necessary
        if self.config['observations']['name'] == "camels_531":
            subset_path = os.path.join(os.getenv("DATA_PATH"), "531sub_id.txt")
            with open(subset_path) as f:
                selected_basins = json.load(f)
            subset_idx = intersect(selected_basins, gage_info)
        else:
            subset_idx = range(len(gage_info))

        x_nn = x_nn[:, subset_idx, :]
        c_nn = c_nn[subset_idx, :]
        target = target[:, subset_idx, :]

        # Convert flow to mm/day if necessary
        target = self._flow_conversion(c_nn, target)

        return x_nn, c_nn, target

    def _flow_conversion(
        self,
        c_nn: NDArray[np.float32],
        target: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Convert hydraulic flow from ft3/s to mm/day.
        
        Parameters
        ----------
        c_nn
            Neural network static data.
        target
            Target variable data.
        """
        for name in ['flow_sim', 'streamflow', 'sf']:
            if name in self.target:
                target_temp = target[:, :, self.target.index(name)]
                area_name = self.config['observations']['area_name']
                basin_area = c_nn[:, self.nn_attributes.index(area_name)]

                area = np.expand_dims(basin_area, axis=0).repeat(
                    target_temp.shape[0], 0
                )
                target[:, :, self.target.index(name)] = (
                    (10 ** 3) * target_temp * 0.0283168 * 3600 * 24
                    / (area * (10 ** 6))
                )
        return target

    def load_norm_stats(
        self,
        x_nn: NDArray[np.float32],
        c_nn: NDArray[np.float32],
        target: NDArray[np.float32],
    ) -> None:
        """Load or calculate normalization statistics if necessary."""
        self.out_path = os.path.join(
            self.config['model_path'],
            'normalization_statistics.json',
        )

        if os.path.isfile(self.out_path) and (not self.overwrite):
            if not self.norm_stats:
                with open(self.out_path) as f:
                    self.norm_stats = json.load(f)
        else:
            # Init normalization stats if file doesn't exist or overwrite is True.
            self.norm_stats = self._init_norm_stats(x_nn, c_nn, target)

    def _init_norm_stats(
        self,
        x_nn: NDArray[np.float32],
        c_nn: NDArray[np.float32],
        target: NDArray[np.float32],
    ) -> dict[str, list[float]]:
        """Compile and save calculations of data normalization statistics.
        
        Parameters
        ----------
        x_nn
            Neural network dynamic data.
        c_nn
            Neural network static data.
        target
            Target variable data.
        
        Returns
        -------
        dict[str, list[float]]
            Dictionary of normalization statistics for each variable.
        """
        stat_dict = {}

        # Get basin areas from attributes.
        basin_area = self._get_basin_area(c_nn)

        # Forcing variable stats
        for k, var in enumerate(self.nn_forcings):
            if var in self.log_norm_vars:
                stat_dict[var] = self._calc_gamma_stats(x_nn[:, :, k])
            else:
                stat_dict[var] = self._calc_norm_stats(x_nn[:, :, k])

        # Attribute variable stats
        for k, var in enumerate(self.nn_attributes):
            stat_dict[var] = self._calc_norm_stats(c_nn[:, k])

        # Target variable stats
        for i, name in enumerate(self.target):
            if name in ['flow_sim', 'streamflow', 'sf']:
                stat_dict[name] = self._calc_norm_stats(
                    np.swapaxes(target[:, :, i:i+1], 1, 0).copy(),
                    basin_area,
                )
            else:
                stat_dict[name] = self._calc_norm_stats(
                    np.swapaxes(target[:, :, i:i+1], 1, 0),
                )

        with open(self.out_path, 'w') as f:
            json.dump(stat_dict, f, indent=4)

        return stat_dict

    def _calc_norm_stats(
        self,
        x: NDArray[np.float32],
        basin_area: NDArray[np.float32] = None,
    ) -> list[float]:
        """Calculate statistics for normalization.

        Parameters
        ----------
        x
            Input data array.
        basin_area
            Basin area array for normalization.
        
        Returns
        -------
        list[float]
            List of statistics [10th percentile, 90th percentile, mean, std].
        """
        # Handle invalid values
        x[x == -999] = np.nan
        if basin_area is not None:
            x[x < 0] = 0

        # Basin area normalization
        if basin_area is not None:
            nd = len(x.shape)
            if (nd == 3) and (x.shape[2] == 1):
                x = x[:, :, 0]
            temparea = np.tile(basin_area, (1, x.shape[1]))
            flow = (x * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10 ** 3
            x = flow

        # Flatten and exclude NaNs and invalid values
        a = x.flatten()
        if basin_area is None:
            if len(x.shape) > 1:
                a = np.swapaxes(x, 1, 0).flatten()
            else:
                a = x.flatten()
        b = a[(~np.isnan(a)) & (a != -999999)]
        if b.size == 0:
            b = np.array([0])

        # Calculate stats
        if basin_area is not None:
            transformed = np.log10(np.sqrt(b) + 0.1)
        else:
            transformed = b
        p10, p90 = np.percentile(transformed, [10, 90]).astype(float)
        mean = np.mean(transformed).astype(float)
        std = np.std(transformed).astype(float)

        return [p10, p90, mean, max(std, 0.001)]

    def _calc_gamma_stats(self, x: NDArray[np.float32]) -> list[float]:
        """Calculate gamma statistics for streamflow and precipitation data.
        
        Parameters
        ----------
        x
            Input data array.
        
        Returns
        -------
        list[float]
            List of statistics [10th percentile, 90th percentile, mean, std].
        """
        a = np.swapaxes(x, 1, 0).flatten()
        b = a[(~np.isnan(a))]
        b = np.log10(np.sqrt(b) + 0.1)

        p10, p90 = np.percentile(b, [10, 90]).astype(float)
        mean = np.mean(b).astype(float)
        std = np.std(b).astype(float)

        return [p10, p90, mean, max(std, 0.001)]

    def _get_basin_area(
        self,
        c_nn: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Get basin area from attributes.
        
        Parameters
        ----------
        c_nn
            Neural network static data.
        
        Returns
        -------
        NDArray[np.float32]
            1D array of basin areas (2nd dummy dim added for calculations).
        """
        try:
            area_name = self.config['observations']['area_name']
            basin_area = c_nn[:, self.nn_attributes.index(area_name)][:, np.newaxis]
        except KeyError:
            log.warning(
                "No 'area_name' in observation config. "
                "Basin area norm will not be applied."
            )
            basin_area = None
        return basin_area

    def normalize(
        self,
        x_nn: NDArray[np.float32],
        c_nn: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Normalize data for neural network.
        
        Parameters
        ----------
        x_nn
            Neural network dynamic data.
        c_nn
            Neural network static data.
        
        Returns
        -------
        tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]
            Tuple of (x_nn_norm, xc_nn_norm, c_nn_norm).
        """
        x_nn_norm = self._to_norm(
            np.swapaxes(x_nn, 1, 0).copy(),
            self.nn_forcings,
        )
        c_nn_norm = self._to_norm(
            c_nn,
            self.nn_attributes,
        )

        # Remove nans
        x_nn_norm[x_nn_norm != x_nn_norm] = 0
        c_nn_norm[c_nn_norm != c_nn_norm] = 0

        c_nn_norm_expand = np.repeat(
            np.expand_dims(c_nn_norm, 0),
            x_nn_norm.shape[0],
            axis=0,
        )

        xc_nn_norm = np.concatenate((x_nn_norm, c_nn_norm_expand), axis=2)
        return x_nn_norm, xc_nn_norm, c_nn_norm

    def _to_norm(
        self,
        data: NDArray[np.float32],
        vars: list[str],
    ) -> NDArray[np.float32]:
        """Standard data normalization.
        
        Parameters
        ----------
        data
            Data to normalize.
        vars
            List of variable names in data to normalize.
        
        Returns
        -------
        NDArray[np.float32]
            Normalized data.
        """
        data_norm = np.zeros(data.shape)

        for k, var in enumerate(vars):
            stat = self.norm_stats[var]

            if len(data.shape) == 3:
                if var in self.log_norm_vars:
                    data[:, :, k] = np.log10(np.sqrt(data[:, :, k]) + 0.1)
                data_norm[:, :, k] = (data[:, :, k] - stat[2]) / stat[3]
            elif len(data.shape) == 2:
                if var in self.log_norm_vars:
                    data[:, k] = np.log10(np.sqrt(data[:, k]) + 0.1)
                data_norm[:, k] = (data[:, k] - stat[2]) / stat[3]
            else:
                raise DataDimensionalityWarning(
                    "Data dimension must be 2 or 3."
                )

        if len(data_norm.shape) < 3:
            return data_norm
        else:
            return np.swapaxes(data_norm, 1, 0)

    def _from_norm(
        self,
        data_scaled: NDArray[np.float32],
        vars: list[str],
    ) -> NDArray[np.float32]:
        """De-normalize data.
        
        Parameters
        ----------
        data_scaled
            Data to de-normalize.
        vars
            List of variable names in data to de-normalize.
        
        Returns
        -------
        NDArray[np.float32]
            De-normalized data.
        """
        data = np.zeros(data_scaled.shape)

        for k, var in enumerate(vars):
            stat = self.norm_stats[var]
            if len(data_scaled.shape) == 3:
                data[:, :, k] = data_scaled[:, :, k] * stat[3] + stat[2]
                if var in self.log_norm_vars:
                    data[:, :, k] = (np.power(10, data[:, :, k]) - 0.1) ** 2
            elif len(data_scaled.shape) == 2:
                data[:, k] = data_scaled[:, k] * stat[3] + stat[2]
                if var in self.log_norm_vars:
                    data[:, k] = (np.power(10, data[:, k]) - 0.1) ** 2
            else:
                raise DataDimensionalityWarning(
                    "Data dimension must be 2 or 3."
                )

        if len(data.shape) < 3:
            return data
        else:
            return np.swapaxes(data, 1, 0)
