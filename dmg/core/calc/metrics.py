import csv
import json
import logging
import os
from typing import Any, Iterable, Optional, Set

import numpy as np
import scipy.stats as stats
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

log = logging.getLogger()


class Metrics(BaseModel):
    """Metrics for model evaluation.

    Using Pydantic BaseModel for validation.
    Metrics are calculated at each grid point and are listed below.
    
    Adapted from Tadd Bindas, Yalan Song, Farshid Rahmani.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    pred: NDArray[np.float32]
    target: NDArray[np.float32]
    metrics_to_compute: Optional[Set[str]] = Field(default=None, exclude=True)

    bias: NDArray[np.float32] = np.ndarray([])
    bias_rel: NDArray[np.float32] = np.ndarray([])

    rmse: NDArray[np.float32] = np.ndarray([])
    rmse_ub: NDArray[np.float32] = np.ndarray([])
    rmse_fdc: NDArray[np.float32] = np.ndarray([])

    mae: NDArray[np.float32] = np.ndarray([])

    corr: NDArray[np.float32] = np.ndarray([])
    corr_spearman: NDArray[np.float32] = np.ndarray([])
    r2: NDArray[np.float32] = np.ndarray([])
    nse: NDArray[np.float32] = np.ndarray([])

    flv: NDArray[np.float32] = np.ndarray([])
    fhv: NDArray[np.float32] = np.ndarray([])
    pbias: NDArray[np.float32] = np.ndarray([])
    pbias_mid: NDArray[np.float32] = np.ndarray([])
    flv_abs: NDArray[np.float32] = np.ndarray([])
    fhv_abs: NDArray[np.float32] = np.ndarray([])
    pbias_abs: NDArray[np.float32] = np.ndarray([])
    pbias_abs_mid: NDArray[np.float32] = np.ndarray([])

    kge: NDArray[np.float32] = np.ndarray([])
    kge_12: NDArray[np.float32] = np.ndarray([])
    inv_kge: NDArray[np.float32] = np.ndarray([])

    rmse_low: NDArray[np.float32] = np.ndarray([])
    rmse_mid: NDArray[np.float32] = np.ndarray([])
    rmse_high: NDArray[np.float32] = np.ndarray([])

    d_max: NDArray[np.float32] = np.ndarray([])
    d_max_rel: NDArray[np.float32] = np.ndarray([])

    def __init__(
        self,
        pred: NDArray[np.float32],
        target: NDArray[np.float32],
        metrics_to_compute: Optional[Iterable[str]] = None,
    ) -> None:
        if pred.ndim == 1:
            pred = np.expand_dims(pred, axis=0)
        if target.ndim == 1:
            target = np.expand_dims(target, axis=0)
        mtc_set = set(metrics_to_compute) if metrics_to_compute is not None else None
        super().__init__(pred=pred, target=target, metrics_to_compute=mtc_set)

    def model_post_init(self, __context: Any) -> Any:
        """Calculate metrics.

        This method is called after the model is initialized.

        Parameters
        ----------
        __context : Any
            Context object.

        Returns
        -------
        Any
            Context object.
        """
        def wants(name: str) -> bool:
            return self.metrics_to_compute is None or name in self.metrics_to_compute

        if wants('bias'):
            self.bias = self._bias(self.pred, self.target, offset=0.00001)
        if wants('bias_rel'):
            self.bias_rel = self._bias_rel(self.pred, self.target)

        if wants('rmse'):
            self.rmse = self._rmse(self.pred, self.target)
        if wants('rmse_ub'):
            self.rmse_ub = self._rmse_ub(self.pred, self.target)
        if wants('rmse_fdc'):
            self.rmse_fdc = self._rmse_fdc(self.pred, self.target)

        if wants('mae'):
            self.mae = self._mae(self.pred, self.target)

        if wants('corr'):
            self.corr = np.full(self.ngrid, np.nan)
        if wants('corr_spearman'):
            self.corr_spearman = np.full(self.ngrid, np.nan)
        if wants('r2') or wants('nse'):
            self.r2 = np.full(self.ngrid, np.nan)
            self.nse = np.full(self.ngrid, np.nan)

        if wants('flv'):
            self.flv = np.full(self.ngrid, np.nan)
        if wants('fhv'):
            self.fhv = np.full(self.ngrid, np.nan)
        if wants('pbias'):
            self.pbias = np.full(self.ngrid, np.nan)
        if wants('pbias_mid'):
            self.pbias_mid = np.full(self.ngrid, np.nan)

        if wants('flv_abs'):
            self.flv_abs = np.full(self.ngrid, np.nan)
        if wants('fhv_abs'):
            self.fhv_abs = np.full(self.ngrid, np.nan)
        if wants('pbias_abs'):
            self.pbias_abs = np.full(self.ngrid, np.nan)
        if wants('pbias_abs_mid'):
            self.pbias_abs_mid = np.full(self.ngrid, np.nan)

        if wants('kge'):
            self.kge = np.full(self.ngrid, np.nan)
        if wants('kge_12'):
            self.kge_12 = np.full(self.ngrid, np.nan)
        if wants('inv_kge'):
            self.inv_kge = np.full(self.ngrid, np.nan)

        if wants('rmse_low'):
            self.rmse_low = np.full(self.ngrid, np.nan)
        if wants('rmse_mid'):
            self.rmse_mid = np.full(self.ngrid, np.nan)
        if wants('rmse_high'):
            self.rmse_high = np.full(self.ngrid, np.nan)

        if wants('d_max'):
            self.d_max = np.full(self.ngrid, np.nan)
        if wants('d_max_rel'):
            self.d_max_rel = np.full(self.ngrid, np.nan)

        for i in range(0, self.ngrid):
            _pred = self.pred[i]
            _target = self.target[i]
            idx = np.where(
                np.logical_and(~np.isnan(_pred), ~np.isnan(_target)),
            )[0]

            if idx.shape[0] > 0:
                pred = _pred[idx]
                target = _target[idx]

                pred_sort = np.sort(pred)
                target_sort = np.sort(target)
                index_low = round(0.3 * pred_sort.shape[0])
                index_high = round(0.98 * pred_sort.shape[0])

                low_pred = pred_sort[:index_low]
                mid_pred = pred_sort[index_low:index_high]
                high_pred = pred_sort[index_high:]

                low_target = target_sort[:index_low]
                mid_target = target_sort[index_low:index_high]
                high_target = target_sort[index_high:]

                if wants('flv'):
                    self.flv[i] = self._pbias(low_pred, low_target, offset=0.0001)
                if wants('fhv'):
                    self.fhv[i] = self._pbias(high_pred, high_target)
                if wants('pbias'):
                    self.pbias[i] = self._pbias(pred, target)
                if wants('pbias_mid'):
                    self.pbias_mid[i] = self._pbias(mid_pred, mid_target)

                if wants('flv_abs'):
                    self.flv_abs[i] = self._pbias_abs(low_pred, low_target, offset=0.0001)
                if wants('fhv_abs'):
                    self.fhv_abs[i] = self._pbias_abs(high_pred, high_target)
                if wants('pbias_abs'):
                    self.pbias_abs[i] = self._pbias_abs(pred, target)
                if wants('pbias_abs_mid'):
                    self.pbias_abs_mid[i] = self._pbias_abs(mid_pred, mid_target)

                if wants('rmse_low'):
                    self.rmse_low[i] = self._rmse(low_pred, low_target, axis=0)
                if wants('rmse_mid'):
                    self.rmse_mid[i] = self._rmse(mid_pred, mid_target, axis=0)
                if wants('rmse_high'):
                    self.rmse_high[i] = self._rmse(high_pred, high_target, axis=0)

                if wants('d_max') or wants('d_max_rel'):
                    target_max = np.nanmax(target)
                    pred_max = self._pred_max(pred, target, lb=10, ub=11)
                    if wants('d_max'):
                        self.d_max[i] = pred_max - target_max
                    if wants('d_max_rel'):
                        self.d_max_rel[i] = (pred_max - target_max) / target_max * 100

                need_corr = any(
                    wants(name) for name in ['corr', 'corr_spearman', 'kge', 'kge_12', 'inv_kge']
                )
                need_nse_r2 = wants('nse') or wants('r2')

                if idx.shape[0] > 1 and (need_corr or need_nse_r2):
                    corr_val = None
                    if need_corr:
                        corr_val = self._corr(pred, target)
                        if wants('corr'):
                            self.corr[i] = corr_val
                        if wants('corr_spearman'):
                            self.corr_spearman[i] = self._corr_spearman(pred, target)

                    _pred_mean = pred.mean()
                    _target_mean = target.mean()
                    _pred_std = np.std(pred)
                    _target_std = np.std(target)

                    if wants('kge'):
                        self.kge[i] = self._kge(
                            _pred_mean, _target_mean, _pred_std, _target_std, corr_val or self._corr(pred, target),
                        )
                    if wants('kge_12'):
                        self.kge_12[i] = self._kge_12(
                            _pred_mean, _target_mean, _pred_std, _target_std, corr_val or self._corr(pred, target),
                        )
                    if wants('inv_kge'):
                        self.inv_kge[i] = self._inv_kge(pred, target)

                    if need_nse_r2:
                        nse_val, r2_val = self._nse_r2(pred, target, _target_mean)
                        if wants('nse'):
                            self.nse[i] = nse_val
                        if wants('r2'):
                            self.r2[i] = r2_val

        return super().model_post_init(__context)

    @model_validator(mode='after')
    @classmethod
    def validate_pred(cls, metrics: Any) -> Any:
        """Checks that there are no NaN predictions.
        
        Parameters
        ----------
        metrics : Any
            Metrics object.

        Raises
        ------
        ValueError
            If there are NaN predictions.
        
        Returns
        -------
        Any
            Metrics object.
        """
        pred = metrics.pred
        if np.isnan(pred).sum() > 0:
            msg = "Pred contains NaN, check your gradient chain"
            log.exception(msg)
            raise ValueError(msg)
        return metrics

    def calc_stats(self, *args, **kwargs) -> dict[str, dict[str, float]]:
        """Calculate aggregate statistics of metrics."""
        stats = {}
        model_dict = self.model_dump()
        model_dict.pop('pred', None)
        model_dict.pop('target', None)

        if self.metrics_to_compute is not None:
            allowed = self.metrics_to_compute
            model_dict = {k: v for k, v in model_dict.items() if k in allowed}

        # Calculate statistics
        for key, value in model_dict.items():
            if isinstance(value, np.ndarray) and value.size > 0:
                stats[key] = {
                    'median': float(np.nanmedian(value)),
                    'mean': float(np.nanmean(value)),
                    'std': float(np.nanstd(value)),
                }
        return stats

    def model_dump_agg_stats(self, path: str) -> None:
        """Dump aggregate statistics (median, mean, std) to json or csv.
        
        Parameters
        ----------
        path : str
            Path to save file.
        """
        stats = self.calc_stats()

        if path.endswith('.json'):
            with open(path, 'w') as f:
                json.dump(stats, f, indent=4)
        elif path.endswith('.csv'):
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Median', 'Mean', 'Std'])
                for metric, values in stats.items():
                    writer.writerow([metric, values['median'], values['mean'], values['std']])
        else:
            raise ValueError("Provide either a .json or .csv file path.")

    def model_dump_json(self, *args, **kwargs) -> str:
        """Dump raw metrics to json."""
        model_dict = self.model_dump()

        # Keep only requested metrics if a subset was specified
        if self.metrics_to_compute is not None:
            allowed = self.metrics_to_compute
            model_dict = {k: v for k, v in model_dict.items() if k in allowed}

        # Convert ndarray to list for JSON serialization
        for key, value in list(model_dict.items()):
            if isinstance(value, np.ndarray):
                model_dict[key] = value.tolist()

        # Drop raw predictions/targets to keep output compact
        model_dict.pop('pred', None)
        model_dict.pop('target', None)

        return json.dumps(model_dict, *args, **kwargs)

    def dump_metrics(self, path: str) -> None:
        """Dump all metrics and aggregate statistics (median, mean, std) to json.
        
        Parameters
        ----------
        path : str
            Path to save file.
        """
        # Save aggregate statistics
        save_path = os.path.join(path, 'metrics_agg.json')
        self.model_dump_agg_stats(save_path)

        # Save raw metrics
        save_path = os.path.join(path, 'metrics.json')
        json_dat = self.model_dump_json(indent=4)

        with open(save_path, "w") as f:
            json.dump(json_dat, f)

    def tile_mean(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        """Calculate mean of target.

        Parameters
        ----------
        data : NDArray[np.float32]
            Data to calculate mean.
        
        Returns
        -------
        NDArray[np.float32]
            Mean of data.
        """
        return np.tile(np.nanmean(data, axis=1), (self.nt, 1)).transpose()

    @property
    def ngrid(self) -> int:
        """Calculate number of items in grid."""
        return self.pred.shape[0]

    @property
    def nt(self) -> int:
        """Calculate number of time steps."""
        return self.pred.shape[1]

    @staticmethod
    def _bias(
        pred: NDArray[np.float32],
        target: NDArray[np.float32],
        offset: float = 0.0,
    ) -> NDArray[np.float32]:
        """Calculate bias."""
        return np.nanmean(abs(pred - target)/(target + offset), axis=1)

    @staticmethod
    def _bias_rel(
        pred: NDArray[np.float32],
        target: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Calculate relative bias.
        
        Don't sum together because if NaNs are present at different idx,
        the sum will be off.
        """
        pred_sum = np.nansum(pred, axis=1)
        target_sum = np.nansum(target, axis=1)
        return (pred_sum - target_sum) / target_sum

    @staticmethod
    def _pbias(
        pred: NDArray[np.float32],
        target: NDArray[np.float32],
        offset: float = 0.0,
    ) -> np.float32:
        """Calculate percent bias with dynamic offset to avoid extreme values."""
        target_sum = np.sum(target)
        # Use dynamic offset: max of 1% of target sum or provided offset
        eps_dyn = max(0.01 * abs(target_sum), offset) if offset > 0 else 0.0
        return np.sum(pred - target) / (target_sum + eps_dyn) * 100

    @staticmethod
    def _pbias_abs(
        pred: NDArray[np.float32],
        target: NDArray[np.float32],
        offset: float = 0.0,
    ) -> np.float32:
        """Calculate absolute percent bias with dynamic offset to avoid extreme values."""
        target_sum = np.sum(target)
        # Use dynamic offset: max of 1% of target sum or provided offset
        eps_dyn = max(0.01 * abs(target_sum), offset) if offset > 0 else 0.0
        return np.sum(abs(pred - target)) / (target_sum + eps_dyn) * 100

    @staticmethod
    def _rmse(
        pred: NDArray[np.float32],
        target: NDArray[np.float32],
        axis: Optional[int] = 1,
    ) -> NDArray[np.float32]:
        """Calculate root mean square error."""
        return np.sqrt(np.nanmean((pred - target) ** 2, axis=axis))

    def _rmse_ub(
        self,
        pred: NDArray[np.float32],
        target: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Calculate unbiased root mean square error."""
        pred_mean = self.tile_mean(self.pred)
        target_mean = self.tile_mean(self.target)
        pred_anom = self.pred - pred_mean
        target_anom = self.target - target_mean
        return self._rmse(pred_anom, target_anom)

    def _rmse_fdc(
        self,
        pred: NDArray[np.float32],
        target: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Calculate flow duration curve root mean square error."""
        pred_fdc = self._calc_fdc(pred)
        target_fdc = self._calc_fdc(target)
        return self._rmse(pred_fdc, target_fdc)

    @staticmethod
    def _mae(
        pred: NDArray[np.float32],
        target: NDArray[np.float32],
        axis: Optional[int] = 1,
    ) -> NDArray[np.float32]:
        """Calculate mean absolute error."""
        return np.nanmean(np.abs(pred - target), axis=axis)

    def _calc_fdc(
        self,
        data: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Calculate flow duration curve for each grid point."""
        fdc_100 = np.full([self.ngrid, 100], np.nan)
        for i in range(self.ngrid):
            data_slice = data[i]
            non_nan_data_slice = data_slice[~np.isnan(data_slice)]

            if len(non_nan_data_slice) == 0:
                non_nan_data_slice = np.full(self.nt, 0)

            sorted_data = np.sort(non_nan_data_slice)[::-1]
            Nlen = len(non_nan_data_slice)
            ind = (np.arange(100) / 100 * Nlen).astype(int)
            fdc_flow = sorted_data[ind]

            if len(fdc_flow) != 100:
                raise Exception("Unknown assimilation variable")
            else:
                fdc_100[i] = fdc_flow

        return fdc_100

    @staticmethod
    def _pred_max(
        pred: NDArray[np.float32],
        target: NDArray[np.float32],
        lb: int = 0,
        ub: int = 10,
    ) -> np.float32:
        """Calculate maximum value of predictions.
        
        Parameters
        ----------
        pred : NDArray[np.float32]
            Predictions.
        target : NDArray[np.float32]
            Target values.
        lb : int, optional
            Lower bound. Default is 0.
        ub : int, optional
            Upper bound. Default is 10.
        """
        idx_max = np.nanargmax(target)
        if (idx_max < lb):
            lb = idx_max
        elif (ub > len(pred) - idx_max):
            ub = len(pred) - idx_max
        else:
            pass
        return np.nanmax(pred[idx_max - lb:idx_max + ub])

    @staticmethod
    def _corr(
        pred: NDArray[np.float32], target: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Calculate correlation."""
        return stats.pearsonr(pred, target)[0]

    @staticmethod
    def _corr_spearman(
        pred: NDArray[np.float32], target: NDArray[np.float32],
    ) -> np.float32:
        """Calculate Spearman correlation."""
        return stats.spearmanr(pred, target)[0]

    @staticmethod
    def _kge(
        pred_mean: np.float32,
        target_mean: np.float32,
        pred_std: np.float32,
        target_std: np.float32,
        corr: np.float32,
    ) -> NDArray[np.float32]:
        """Calculate Kling-Gupta Efficiency (KGE)."""
        kge = 1 - np.sqrt(
            (corr - 1) ** 2
            + (pred_std / target_std - 1) ** 2
            + (pred_mean / target_mean - 1) ** 2,
        )
        return kge

    @staticmethod
    def _kge_12(
        pred_mean: np.float32,
        target_mean: np.float32,
        pred_std: np.float32,
        target_std: np.float32,
        corr: np.float32,
    ) -> NDArray[np.float32]:
        """Calculate Kling-Gupta Efficiency (KGE) 1-2."""
        kge_12 = 1 - np.sqrt(
            (corr - 1) ** 2
            + ((pred_std * target_mean) / (target_std * pred_mean) - 1) ** 2
            + (pred_mean / target_mean - 1) ** 2,
        )
        return kge_12

    @staticmethod
    def _inv_kge(
        pred: NDArray[np.float32],
        target: NDArray[np.float32],
        stability_eps: float = 1e-6,
    ) -> np.float32:
        """Calculate inverse KGE following the inverse-flow formulation.

        Steps mirror KgeInverseLoss:
        - epsilon = max(0.01 * mean(target), 1e-3)
        - transform: 1 / (flow + epsilon)
        - apply standard KGE on transformed values with small stability eps
        """
        # clamp to non-negative for safety
        if pred.size == 0 or target.size == 0:
            return np.nan

        # clamp to non-negative for safety
        pred_safe = np.clip(pred, a_min=0.0, a_max=None)
        target_safe = np.clip(target, a_min=0.0, a_max=None)

        # 1% 均值法则，但不仅限于此
        eps_dyn = max(0.01 * float(np.mean(target_safe)), 1e-3)

        pred_inv = 1.0 / (pred_safe + eps_dyn)
        target_inv = 1.0 / (target_safe + eps_dyn)

        if pred_inv.size < 2:
            return np.nan

        mean_p = float(np.mean(pred_inv))
        mean_t = float(np.mean(target_inv))
        std_p = float(np.std(pred_inv))
        std_t = float(np.std(target_inv))

        dev_p = pred_inv - mean_p
        dev_t = target_inv - mean_t
        numerator = float(np.sum(dev_p * dev_t))
        denom = float(np.sqrt(np.sum(dev_p ** 2)) * np.sqrt(np.sum(dev_t ** 2)) + stability_eps)
        r = numerator / denom if denom != 0 else 0.0

        beta = mean_p / (mean_t + stability_eps)
        gamma = std_p / (std_t + stability_eps)

        inv_kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2)
        return inv_kge

    @staticmethod
    def _nse_r2(
        pred: NDArray[np.float32],
        target: NDArray[np.float32],
        target_mean: np.float32,
    ) -> tuple[np.float32, np.float32]:
        """Calculate Nash-Sutcliffe Efficiency (NSE) and R^2 (identical here)."""
        sst = np.sum((target - target_mean) ** 2)
        ssres = np.sum((target - pred) ** 2)
        nse_val = np.float32(1 - ssres / sst)
        return nse_val, nse_val
