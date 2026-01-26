"""统计不同 batch size / pred len 的更新速度指标。

指标定义
--------
1) Optimization Step Latency (T_step)
   单次参数更新耗时（含 forward + loss + backward + optimizer step）。
   对 GPU 并行训练，使用 T_step = T_batch / B。

2) Effective Throughput (R_eff)
   使用 TBPTT 时，按 k = ceil(L_total / L_win) 归一化：
   R_eff = B / (T_batch * k) = 1 / (T_step * k)

本文件提供：
- 复刻的 n_iter_ep 计算（与 create_training_grid 一致）
- 按批次耗时计算 T_step / R_eff
- 支持 CSV 读写（列：batch_size, pred_len, t_batch）
"""

from __future__ import annotations

import csv
import logging
import math
import re
import numpy as np
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

#! HBV 730 200 缺数据

@dataclass(frozen=True)
class TimingRecord:
	batch_size: int
	pred_len: int
	t_batch: float  # seconds
	model: str | None = None

def time_to_date(t: int, hr: bool = False) -> Union[dt.date, dt.datetime]:
    """Convert time to date or datetime object.
    
    Adapted from Farshid Rahmani.
    
    Parameters
    ----------
    t
        Time object to convert.
    hr
        If True, return datetime object.
    
    Returns
    -------
    Union[dt.date, dt.datetime]
        The converted date or datetime object.
    """
    tOut = None

    if type(t) is str:
        t = int(t.replace('/', ''))

    if type(t) is int:
        if t < 30000000 and t > 10000000:
            t = dt.datetime.strptime(str(t), "%Y%m%d").date()
            tOut = t if hr is False else t.datetime()

    if type(t) is dt.date:
        tOut = t if hr is False else t.datetime()

    if type(t) is dt.datetime:
        tOut = t.date() if hr is False else t

    if tOut is None:
        raise Exception("Failed to change time to date.")
    return tOut

def trange_to_array(
    t_range: np.NDArray[np.float32],
    *,
    step=np.timedelta64(1, "D"),
) -> np.NDArray[np.float32]:
    """Convert time range to array of dates.
    
    Parameters
    ----------
    t_range
        Time range to convert.
    step
        Step size for the array.
    
    Returns
    -------
    NDArray[np.float32]
        Array of dates.
    """
    sd = time_to_date(t_range[0])
    ed = time_to_date(t_range[1])
    return np.arange(sd, ed, step)

def compute_training_grid(
	*,
	n_t: int,
	n_samples: int,
	batch_size: int,
	rho: int,
	warm_up: int = 0,
	train_time: tuple[Any, Any] | None = None,
) -> tuple[int, int, int]:
	"""复刻 create_training_grid，用于计算 n_iter_ep。

	Parameters
	----------
	n_t
		总时间步数（例如 9 年逐日数据长度）。
	n_samples
		样本数（如流域数量）。
	batch_size
		训练 batch size。
	rho
		预测长度（pred len）。
	warm_up
		暖启动长度（默认 365）。
	train_time
		训练时间范围 (start, end)，用于限制 rho。

	Returns
	-------
	tuple[int, int, int]
		n_samples, n_iter_ep, n_t
	"""
	if train_time is not None:
		t = trange_to_array(train_time)
		rho = min(t.shape[0], rho)

	denom = 1 - (batch_size * rho) / n_samples / (n_t - warm_up)
	if denom <= 0:
		raise ValueError(
			"Invalid denominator when computing n_iter_ep. "
			"Check batch_size, rho, n_samples, n_t, warm_up."
		)

	n_iter_ep = int(math.ceil(math.log(0.01) / math.log(denom)))
	return n_samples, n_iter_ep, n_t


def compute_efficiency_metrics(
	*,
	batch_size: int,
	pred_len: int,
	t_batch: float,
	l_total: int,
) -> dict[str, float]:
	"""计算 T_step 与 R_eff。"""
	if batch_size <= 0:
		raise ValueError("batch_size must be > 0")
	if pred_len <= 0:
		raise ValueError("pred_len must be > 0")
	if t_batch <= 0:
		raise ValueError("t_batch must be > 0")
	if l_total <= 0:
		raise ValueError("l_total must be > 0")

	t_step = t_batch / batch_size
	k = int(math.ceil(l_total / pred_len))
	r_eff = batch_size / (t_batch * k)
	return {
		"T_step": t_step,
		"k": float(k),
		"R_eff": r_eff,
	}


def load_csv(path: Path) -> list[TimingRecord]:
	"""读取 CSV，列名需包含 batch_size, pred_len, t_batch。"""
	rows: list[TimingRecord] = []
	with path.open("r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			rows.append(
				TimingRecord(
					batch_size=int(row["batch_size"]),
					pred_len=int(row["pred_len"]),
					t_batch=float(row["t_batch"]),
					model=row.get("model"),
				)
			)
	return rows


def _parse_config_dir(name: str) -> tuple[int, int] | None:
	"""从目录名解析 pred_len 与 batch_size（Rxxx, Bxx）。"""
	match = re.search(r"_R(\d+)_B(\d+)_", name)
	if not match:
		return None
	return int(match.group(1)), int(match.group(2))


def _epoch_seconds_from_stat_dir(stat_dir: Path) -> float | None:
	"""通过 Ep1/Ep2 文件 mtime 计算单个 epoch 时间（秒）。"""
	ep1 = stat_dir / "dUnifyV2_Ep1.pt"
	ep2 = stat_dir / "dUnifyV2_Ep2.pt"
	if not ep1.exists() or not ep2.exists():
		return None
	return abs(ep2.stat().st_mtime - ep1.stat().st_mtime)


def scan_base_path(base_path: Path) -> list[TimingRecord]:
	"""遍历基础路径生成 TimingRecord（从 Ep1/Ep2 时间差得到 t_batch）。"""
	records: list[TimingRecord] = []
	for config_dir in sorted(p for p in base_path.iterdir() if p.is_dir()):
		parsed = _parse_config_dir(config_dir.name)
		if not parsed:
			continue
		pred_len, batch_size = parsed

		for model_dir in sorted(p for p in config_dir.iterdir() if p.is_dir()):
			stat_dir = model_dir / "KgeLoss" / "stat"
			if not stat_dir.is_dir():
				continue
			elapsed = _epoch_seconds_from_stat_dir(stat_dir)
			if elapsed is None:
				log.warning("Missing Ep1/Ep2 in %s", stat_dir)
				continue
			records.append(
				TimingRecord(
					batch_size=batch_size,
					pred_len=pred_len,
					t_batch=elapsed,  # 这里先存 epoch 时间，后续转为 batch
					model=model_dir.name,
				)
			)
	return records


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
	rows = list(rows)
	if not rows:
		return
	with path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
		writer.writeheader()
		writer.writerows(rows)


def summarize(
	*,
	records: list[TimingRecord],
	n_t: int,
	n_samples: int,
	warm_up: int,
	train_time: tuple[Any, Any] | None,
) -> list[dict[str, Any]]:
	"""生成统计结果表。"""
	rows: list[dict[str, Any]] = []
	l_total = n_t - warm_up
	for r in records:
		_, n_iter_ep, _ = compute_training_grid(
			n_t=n_t,
			n_samples=n_samples,
			batch_size=r.batch_size,
			rho=r.pred_len,
			warm_up=warm_up,
			train_time=train_time,
		)

		# r.t_batch 目前是 epoch 总耗时（秒），转为 batch 耗时
		if n_iter_ep <= 0:
			log.warning("Invalid n_iter_ep for %s", r)
			continue
		t_batch = r.t_batch / n_iter_ep

		metrics = compute_efficiency_metrics(
			batch_size=r.batch_size,
			pred_len=r.pred_len,
			t_batch=t_batch,
			l_total=l_total,
		)
		rows.append(
			{
				"model": r.model,
				"batch_size": r.batch_size,
				"pred_len": r.pred_len,
				"t_epoch": r.t_batch,
				"n_iter_ep": n_iter_ep,
				"t_batch": t_batch,
				"T_step": metrics["T_step"],
				"k": metrics["k"],
				"R_eff": metrics["R_eff"],
			}
		)
	return rows


BASE_PATH = Path(
	"/workspace/my_deltamodel/project/diff_compare/ablation/batchsize/"
	"camels_559/train1989-1998/no_multi"
)
OUTPUT_CSV = Path(
	"/workspace/my_deltamodel/project/diff_compare/plots/csv/"
	"stats_calc_time.csv"
)

# 基本参数（请按你的数据实际情况调整）
N_T = 365 * 9  # 9年逐日
N_SAMPLES = 559
WARM_UP = 365
TRAIN_TIME = ("1989/01/01", "1998/01/01")


def main() -> None:
	records = scan_base_path(BASE_PATH)
	rows = summarize(
		records=records,
		n_t=N_T,
		n_samples=N_SAMPLES,
		warm_up=WARM_UP,
		train_time=TRAIN_TIME,
	)
	write_csv(OUTPUT_CSV, rows)


main()
