"""
Hybrid MIMIC-IV ECG feature extraction pipeline (192 validated features).

Extracts:
- 99 NeuroKit2 features using validated extraction functions
- 93 pre-computed ECGDeli features loaded from CSV/Parquet/DuckDB

Output:
- data/processed/mimic_hybrid_192_features.parquet
- data/processed/mimic_hybrid_192_features.csv
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import partial
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional
import warnings

import duckdb
import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


warnings.filterwarnings(
    "ignore",
    message="A value is being set on a copy of a DataFrame or Series through chained assignment using an inplace method.*",
    module=r"neurokit2\..*",
)


_GPU_STATS_AVAILABLE = False
_GPU_DEVICE = None
_GPU_RESERVED_TENSOR = None
_GPU_RESERVE_DONE = False
_TORCH_MODULE = None
_TORCH_IMPORT_ATTEMPTED = False
MIN_GPU_STATS_SIZE = 1000
ROLLING_CHECKPOINT = "hybrid_192_rolling.parquet"
DELTA_CHECKPOINT_PATTERN = re.compile(r"^hybrid_192_delta_(\d+)_(\d+)\.parquet$")


def _get_torch_module():
    global _TORCH_MODULE, _TORCH_IMPORT_ATTEMPTED
    if _TORCH_IMPORT_ATTEMPTED:
        return _TORCH_MODULE

    _TORCH_IMPORT_ATTEMPTED = True
    try:
        import torch as torch_module
    except Exception:
        _TORCH_MODULE = None
    else:
        _TORCH_MODULE = torch_module
    return _TORCH_MODULE


def _torch_cuda_available() -> bool:
    torch_module = _get_torch_module()
    return torch_module is not None and torch_module.cuda.is_available()


def _is_cuda_oom_error(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "cuda out of memory" in str(exc).lower()


def _gpu_strict_oom_enabled() -> bool:
    return os.environ.get("HYBRID_GPU_STRICT_OOM", "1") == "1"


def _rethrow_if_cuda_oom(exc: BaseException) -> None:
    if _gpu_strict_oom_enabled() and _is_cuda_oom_error(exc):
        raise RuntimeError(
            "CUDA out of memory during feature extraction. "
            "Disable strict failure with --gpu-allow-cpu-fallback if you want automatic CPU fallback."
        ) from exc


def _run_gpu_oom_probe(device_name: str) -> None:
    torch_module = _get_torch_module()
    if torch_module is None or not torch_module.cuda.is_available():
        raise RuntimeError("--gpu-oom-probe requested, but CUDA/PyTorch is not available.")

    device = torch_module.device(device_name)
    free_bytes, total_bytes = torch_module.cuda.mem_get_info(device)
    attempt_bytes = int(free_bytes + (64 * 1024 * 1024))

    print(
        "Running CUDA OOM probe "
        f"(device={device_name}, free_mb={free_bytes / 1024**2:.1f}, total_mb={total_bytes / 1024**2:.1f})"
    )

    try:
        _probe_tensor = torch_module.empty(attempt_bytes, dtype=torch_module.uint8, device=device)
        del _probe_tensor
        torch_module.cuda.empty_cache()
    except RuntimeError as exc:
        if _is_cuda_oom_error(exc):
            raise RuntimeError(
                "CUDA out of memory (intentional probe): "
                f"requested={attempt_bytes / 1024**2:.1f}MB, free={free_bytes / 1024**2:.1f}MB, "
                f"total={total_bytes / 1024**2:.1f}MB"
            ) from exc
        raise

    raise RuntimeError(
        "OOM probe unexpectedly succeeded. "
        "Try running concurrent GPU workloads or increase probe pressure."
    )


def _initialize_gpu_reservation_if_needed() -> None:
    global _GPU_RESERVED_TENSOR, _GPU_RESERVE_DONE
    if _GPU_RESERVE_DONE:
        return

    _GPU_RESERVE_DONE = True

    torch_module = _get_torch_module()
    if torch_module is None or not torch_module.cuda.is_available():
        return

    reserve_mb = int(os.environ.get("HYBRID_GPU_RESERVE_MB", "0") or "0")
    reserve_fraction = float(os.environ.get("HYBRID_GPU_RESERVE_FRACTION", "0") or "0")
    if reserve_mb <= 0 and reserve_fraction <= 0:
        return

    device_name = os.environ.get("HYBRID_GPU_DEVICE", "cuda:0")
    device = torch_module.device(device_name)
    free_bytes, _ = torch_module.cuda.mem_get_info(device)

    target_bytes = max(0, reserve_mb) * 1024 * 1024
    if reserve_fraction > 0:
        target_bytes = max(target_bytes, int(free_bytes * reserve_fraction))

    if target_bytes <= 0:
        return

    try:
        _GPU_RESERVED_TENSOR = torch_module.empty(target_bytes, dtype=torch_module.uint8, device=device)
        print(
            f"GPU reservation active on {device_name}: {target_bytes / 1024**2:.1f} MB"
        )
    except RuntimeError as exc:
        _rethrow_if_cuda_oom(exc)
        if os.environ.get("HYBRID_GPU_STRICT_OOM", "1") == "0":
            _GPU_RESERVED_TENSOR = None
            torch_module.cuda.empty_cache()


LEAD_MAP = {
    "I": 0,
    "II": 1,
    "III": 2,
    "aVR": 3,
    "aVL": 4,
    "aVF": 5,
    "V1": 6,
    "V2": 7,
    "V3": 8,
    "V4": 9,
    "V5": 10,
    "V6": 11,
}


NEUROKIT2_FEATURES = [
    "R_Amp_V1", "R_Amp_V1_count", "R_Amp_V1_iqr",
    "R_Amp_V2", "R_Amp_V2_count", "R_Amp_V2_iqr",
    "R_Amp_aVR", "R_Amp_aVR_count", "R_Amp_aVR_iqr",
    "S_Amp_I", "S_Amp_I_count", "S_Amp_I_iqr",
    "S_Amp_II", "S_Amp_II_count", "S_Amp_II_iqr",
    "S_Amp_III", "S_Amp_III_count", "S_Amp_III_iqr",
    "S_Amp_aVF", "S_Amp_aVF_count", "S_Amp_aVF_iqr",
    "S_Amp_aVL", "S_Amp_aVL_count", "S_Amp_aVL_iqr",
    "S_Amp_aVR", "S_Amp_aVR_count", "S_Amp_aVR_iqr",
    "S_Amp_V1", "S_Amp_V1_count", "S_Amp_V1_iqr",
    "S_Amp_V2", "S_Amp_V2_count", "S_Amp_V2_iqr",
    "S_Amp_V3", "S_Amp_V3_count", "S_Amp_V3_iqr",
    "S_Amp_V4", "S_Amp_V4_count", "S_Amp_V4_iqr",
    "Q_Amp_I", "Q_Amp_I_count", "Q_Amp_I_iqr",
    "Q_Amp_II", "Q_Amp_II_count", "Q_Amp_II_iqr",
    "Q_Amp_III", "Q_Amp_III_count", "Q_Amp_III_iqr",
    "Q_Amp_aVR", "Q_Amp_aVR_count", "Q_Amp_aVR_iqr",
    "Q_Amp_aVL", "Q_Amp_aVL_count", "Q_Amp_aVL_iqr",
    "Q_Amp_aVF", "Q_Amp_aVF_count", "Q_Amp_aVF_iqr",
    "Q_Amp_V1", "Q_Amp_V1_count", "Q_Amp_V1_iqr",
    "Q_Amp_V2", "Q_Amp_V2_count", "Q_Amp_V2_iqr",
    "Q_Amp_V3", "Q_Amp_V3_count", "Q_Amp_V3_iqr",
    "Q_Amp_V4", "Q_Amp_V4_count", "Q_Amp_V4_iqr",
    "Q_Amp_V5", "Q_Amp_V5_count", "Q_Amp_V5_iqr",
    "Q_Amp_V6", "Q_Amp_V6_count", "Q_Amp_V6_iqr",
    "ST_Elev_I", "ST_Elev_I_count", "ST_Elev_I_iqr",
    "ST_Elev_aVL", "ST_Elev_aVL_count", "ST_Elev_aVL_iqr",
    "ST_Elev_V1", "ST_Elev_V1_count", "ST_Elev_V1_iqr",
    "ST_Elev_V2", "ST_Elev_V2_count", "ST_Elev_V2_iqr",
    "ST_Elev_V3", "ST_Elev_V3_count", "ST_Elev_V3_iqr",
    "ST_Elev_V4", "ST_Elev_V4_count", "ST_Elev_V4_iqr",
    "ST_Elev_V5", "ST_Elev_V5_count", "ST_Elev_V5_iqr",
    "ST_Elev_V6", "ST_Elev_V6_count", "ST_Elev_V6_iqr",
]

ECGDELI_FEATURES = [
    "R_Amp_I", "R_Amp_I_count", "R_Amp_I_iqr",
    "R_Amp_II", "R_Amp_II_count", "R_Amp_II_iqr",
    "R_Amp_III", "R_Amp_III_count", "R_Amp_III_iqr",
    "R_Amp_V3", "R_Amp_V3_count", "R_Amp_V3_iqr",
    "R_Amp_V4", "R_Amp_V4_count", "R_Amp_V4_iqr",
    "R_Amp_V5", "R_Amp_V5_count", "R_Amp_V5_iqr",
    "R_Amp_V6", "R_Amp_V6_count", "R_Amp_V6_iqr",
    "R_Amp_aVF", "R_Amp_aVF_count", "R_Amp_aVF_iqr",
    "R_Amp_aVL", "R_Amp_aVL_count", "R_Amp_aVL_iqr",
    "S_Amp_V5", "S_Amp_V5_count", "S_Amp_V5_iqr",
    "S_Amp_V6", "S_Amp_V6_count", "S_Amp_V6_iqr",
    "T_Amp_I", "T_Amp_I_count", "T_Amp_I_iqr",
    "T_Amp_II", "T_Amp_II_count", "T_Amp_II_iqr",
    "T_Amp_III", "T_Amp_III_count", "T_Amp_III_iqr",
    "T_Amp_aVR", "T_Amp_aVR_count", "T_Amp_aVR_iqr",
    "T_Amp_aVL", "T_Amp_aVL_count", "T_Amp_aVL_iqr",
    "T_Amp_aVF", "T_Amp_aVF_count", "T_Amp_aVF_iqr",
    "T_Amp_V1", "T_Amp_V1_count", "T_Amp_V1_iqr",
    "T_Amp_V2", "T_Amp_V2_count", "T_Amp_V2_iqr",
    "T_Amp_V3", "T_Amp_V3_count", "T_Amp_V3_iqr",
    "T_Amp_V4", "T_Amp_V4_count", "T_Amp_V4_iqr",
    "T_Amp_V5", "T_Amp_V5_count", "T_Amp_V5_iqr",
    "T_Amp_V6", "T_Amp_V6_count", "T_Amp_V6_iqr",
    "ST_Elev_II", "ST_Elev_II_count", "ST_Elev_II_iqr",
    "ST_Elev_III", "ST_Elev_III_count", "ST_Elev_III_iqr",
    "ST_Elev_aVR", "ST_Elev_aVR_count", "ST_Elev_aVR_iqr",
    "ST_Elev_aVF", "ST_Elev_aVF_count", "ST_Elev_aVF_iqr",
    "RR_Mean_Global", "RR_Mean_Global_count", "RR_Mean_Global_iqr",
    "QRS_Dur_Global", "QRS_Dur_Global_count", "QRS_Dur_Global_iqr",
    "QT_IntFramingham_Global", "QT_IntFramingham_Global_count", "QT_IntFramingham_Global_iqr",
    "PR_Int_Global", "PR_Int_Global_count", "PR_Int_Global_iqr",
]


def _summary(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"value": np.nan, "count": 0, "iqr": np.nan}

    arr = np.asarray(values, dtype=float)

    use_gpu_stats = (
        os.environ.get("HYBRID_GPU_STATS", "0") == "1"
        and arr.size >= MIN_GPU_STATS_SIZE
    )
    if use_gpu_stats and _torch_cuda_available():
        try:
            _initialize_gpu_reservation_if_needed()
            device_name = os.environ.get("HYBRID_GPU_DEVICE", "cuda:0")
            torch_module = _get_torch_module()
            if torch_module is None:
                raise RuntimeError("GPU stats requested but torch is unavailable.")
            tensor = torch_module.as_tensor(arr, dtype=torch_module.float32, device=device_name)
            q75 = torch_module.quantile(tensor, 0.75)
            q25 = torch_module.quantile(tensor, 0.25)
            return {
                "value": float(torch_module.mean(tensor).item()),
                "count": int(tensor.numel()),
                "iqr": float((q75 - q25).item()),
            }
        except Exception as exc:
            _rethrow_if_cuda_oom(exc)
            pass

    return {
        "value": float(np.mean(arr)),
        "count": int(arr.size),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
    }


def _build_neurokit_lead_context(
    ecg_signal: np.ndarray,
    lead_idx: int,
    sampling_rate: int = 500,
) -> Optional[Dict[str, object]]:
    try:
        lead_signal = ecg_signal[:, lead_idx]
        ecg_cleaned = nk.ecg_clean(lead_signal, sampling_rate=sampling_rate)
        _, rpeaks_dict = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        r_peaks = rpeaks_dict.get("ECG_R_Peaks", [])

        if len(r_peaks) == 0:
            return {
                "lead_signal": lead_signal,
                "r_peaks": r_peaks,
                "waves_dict": {},
                "delineation_ok": True,
                "peaks_ok": False,
            }

        try:
            _, waves_dict = nk.ecg_delineate(ecg_cleaned, rpeaks_dict, sampling_rate=sampling_rate)
            delineation_ok = True
        except Exception:
            waves_dict = {}
            delineation_ok = False

        return {
            "lead_signal": lead_signal,
            "r_peaks": r_peaks,
            "waves_dict": waves_dict,
            "delineation_ok": delineation_ok,
            "peaks_ok": True,
        }
    except Exception as exc:
        _rethrow_if_cuda_oom(exc)
        return None


def calculate_r_amplitude_neurokit(
    ecg_signal: np.ndarray,
    lead_idx: int,
    sampling_rate: int = 500,
    lead_context: Optional[Dict[str, object]] = None,
) -> Dict[str, Optional[float]]:
    try:
        context = lead_context or _build_neurokit_lead_context(ecg_signal, lead_idx, sampling_rate=sampling_rate)
        if context is None:
            return {"value": np.nan, "count": 0, "iqr": np.nan}

        lead_signal = context.get("lead_signal")
        r_peaks = context.get("r_peaks", [])
        waves_dict = context.get("waves_dict", {})
        peaks_ok = bool(context.get("peaks_ok", len(r_peaks) > 0))

        if not peaks_ok or len(r_peaks) == 0:
            return {"value": np.nan, "count": 0, "iqr": np.nan}

        amplitudes = []
        for i, r_peak in enumerate(r_peaks):
            baseline = 0.0
            if "ECG_P_Onsets" in waves_dict and i < len(waves_dict["ECG_P_Onsets"]):
                p_onset = waves_dict["ECG_P_Onsets"][i]
                if p_onset is not None:
                    try:
                        if not np.isnan(p_onset):
                            p_idx = int(p_onset)
                            if 0 <= p_idx < len(lead_signal):
                                baseline = lead_signal[p_idx]
                    except (TypeError, ValueError):
                        pass
            if 0 <= r_peak < len(lead_signal):
                amplitudes.append(float(lead_signal[r_peak] - baseline))

        return _summary(amplitudes)
    except Exception as exc:
        _rethrow_if_cuda_oom(exc)
        return {"value": np.nan, "count": 0, "iqr": np.nan}


def calculate_s_amplitude_neurokit(
    ecg_signal: np.ndarray,
    lead_idx: int,
    sampling_rate: int = 500,
    lead_context: Optional[Dict[str, object]] = None,
) -> Dict[str, Optional[float]]:
    try:
        context = lead_context or _build_neurokit_lead_context(ecg_signal, lead_idx, sampling_rate=sampling_rate)
        if context is None:
            return {"value": np.nan, "count": 0, "iqr": np.nan}

        lead_signal = context.get("lead_signal")
        r_peaks = context.get("r_peaks", [])
        waves_dict = context.get("waves_dict", {})
        delineation_ok = bool(context.get("delineation_ok", False))

        if len(r_peaks) == 0:
            return {"value": np.nan, "count": 0, "iqr": np.nan}
        if not delineation_ok:
            return {"value": np.nan, "count": 0, "iqr": np.nan}

        s_peaks = waves_dict.get("ECG_S_Peaks", [])
        amplitudes = []
        for i, s_peak in enumerate(s_peaks):
            if s_peak is None:
                continue
            try:
                if np.isnan(s_peak):
                    continue
            except TypeError:
                continue

            s_idx = int(s_peak)
            if not (0 <= s_idx < len(lead_signal)):
                continue

            baseline = 0.0
            if "ECG_P_Onsets" in waves_dict and i < len(waves_dict["ECG_P_Onsets"]):
                p_onset = waves_dict["ECG_P_Onsets"][i]
                if p_onset is not None:
                    try:
                        if not np.isnan(p_onset):
                            p_idx = int(p_onset)
                            if 0 <= p_idx < len(lead_signal):
                                baseline = lead_signal[p_idx]
                    except (TypeError, ValueError):
                        pass

            amplitudes.append(float(lead_signal[s_idx] - baseline))

        return _summary(amplitudes)
    except Exception as exc:
        _rethrow_if_cuda_oom(exc)
        return {"value": np.nan, "count": 0, "iqr": np.nan}


def calculate_q_amplitude_neurokit(
    ecg_signal: np.ndarray,
    lead_idx: int,
    sampling_rate: int = 500,
    lead_context: Optional[Dict[str, object]] = None,
) -> Dict[str, Optional[float]]:
    try:
        context = lead_context or _build_neurokit_lead_context(ecg_signal, lead_idx, sampling_rate=sampling_rate)
        if context is None:
            return {"value": np.nan, "count": 0, "iqr": np.nan}

        lead_signal = context.get("lead_signal")
        r_peaks = context.get("r_peaks", [])
        waves_dict = context.get("waves_dict", {})
        delineation_ok = bool(context.get("delineation_ok", False))

        if len(r_peaks) == 0:
            return {"value": np.nan, "count": 0, "iqr": np.nan}
        if not delineation_ok:
            return {"value": np.nan, "count": 0, "iqr": np.nan}

        q_amplitudes = []
        for i in range(len(r_peaks)):
            baseline = 0.0
            if "ECG_P_Onsets" in waves_dict:
                p_onsets = waves_dict["ECG_P_Onsets"]
                if i < len(p_onsets):
                    try:
                        p_val = p_onsets[i]
                        if p_val is not None and not pd.isna(p_val) and np.isfinite(p_val):
                            p_onset_idx = int(p_val)
                            if 0 <= p_onset_idx < len(lead_signal):
                                baseline = lead_signal[p_onset_idx]
                    except (ValueError, TypeError, OverflowError):
                        pass

            if "ECG_Q_Peaks" in waves_dict:
                q_peaks = waves_dict["ECG_Q_Peaks"]
                if i < len(q_peaks):
                    try:
                        q_val = q_peaks[i]
                        if q_val is not None and not pd.isna(q_val) and np.isfinite(q_val):
                            q_peak_idx = int(q_val)
                            if 0 <= q_peak_idx < len(lead_signal):
                                q_amplitude = lead_signal[q_peak_idx] - baseline
                                q_amplitudes.append(float(q_amplitude))
                    except (ValueError, TypeError, OverflowError):
                        pass

        return _summary(q_amplitudes)

    except Exception as exc:
        _rethrow_if_cuda_oom(exc)
        return {"value": np.nan, "count": 0, "iqr": np.nan}


def calculate_st_elevation_neurokit(
    ecg_signal: np.ndarray,
    lead_idx: int,
    sampling_rate: int = 500,
    st_offset_ms: int = 60,
    lead_context: Optional[Dict[str, object]] = None,
) -> Dict[str, Optional[float]]:
    try:
        context = lead_context or _build_neurokit_lead_context(ecg_signal, lead_idx, sampling_rate=sampling_rate)
        if context is None:
            return {"value": np.nan, "count": 0, "iqr": np.nan}

        lead_signal = context.get("lead_signal")
        r_peaks = context.get("r_peaks", [])
        waves_dict = context.get("waves_dict", {})
        delineation_ok = bool(context.get("delineation_ok", False))

        if len(r_peaks) == 0:
            return {"value": np.nan, "count": 0, "iqr": np.nan}
        if not delineation_ok:
            return {"value": np.nan, "count": 0, "iqr": np.nan}

        st_elevations = []

        for i in range(len(r_peaks)):
            j_point_idx = None

            if "ECG_R_Offsets" in waves_dict:
                r_offsets = waves_dict["ECG_R_Offsets"]
                if i < len(r_offsets) and r_offsets[i] is not None:
                    try:
                        if not np.isnan(r_offsets[i]):
                            j_point_idx = int(r_offsets[i])
                    except (ValueError, TypeError):
                        pass

            if j_point_idx is None and "ECG_S_Peaks" in waves_dict:
                s_peaks = waves_dict["ECG_S_Peaks"]
                if i < len(s_peaks) and s_peaks[i] is not None:
                    try:
                        if not np.isnan(s_peaks[i]):
                            s_peak_idx = int(s_peaks[i])
                            j_point_idx = s_peak_idx + int(0.02 * sampling_rate)
                    except (ValueError, TypeError):
                        pass

            if j_point_idx is None:
                estimated_qrs_ms = 90
                estimated_qrs = int(estimated_qrs_ms / 1000 * sampling_rate)
                j_point_idx = r_peaks[i] + estimated_qrs

            if not (0 <= j_point_idx < len(lead_signal)):
                continue

            baseline = 0.0
            if "ECG_P_Onsets" in waves_dict:
                p_onsets = waves_dict["ECG_P_Onsets"]
                if i < len(p_onsets) and p_onsets[i] is not None:
                    try:
                        if not np.isnan(p_onsets[i]):
                            p_onset_idx = int(p_onsets[i])
                            if 0 <= p_onset_idx < len(lead_signal):
                                baseline = lead_signal[p_onset_idx]
                    except (ValueError, TypeError):
                        pass

            st_measurement_point = j_point_idx + int(st_offset_ms / 1000 * sampling_rate)
            if not (0 <= st_measurement_point < len(lead_signal)):
                continue

            st_value = lead_signal[st_measurement_point] - baseline
            st_elevations.append(float(st_value))

        return _summary(st_elevations)
    except Exception as exc:
        _rethrow_if_cuda_oom(exc)
        return {"value": np.nan, "count": 0, "iqr": np.nan}


def load_mimic_record_index(
    db_path: Path,
    cohort_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    con = duckdb.connect(str(db_path))
    try:
        base_select = """
        SELECT
            rl.subject_id,
            rl.study_id,
            rl.ecg_time,
            COALESCE(wnl.waveform_path, rl.path || CAST(rl.file_name AS VARCHAR)) AS waveform_path
        FROM record_list rl
        LEFT JOIN waveform_note_links wnl
            ON rl.subject_id = wnl.subject_id
           AND rl.study_id = wnl.study_id
        """

        if cohort_path is not None and cohort_path.exists():
            query = f"""
            WITH cohort AS (
                SELECT DISTINCT subject_id, study_id, hadm_id, primary_label, index_mi_time, hours_from_mi
                FROM read_parquet('{str(cohort_path).replace("'", "''")}')
            )
            SELECT b.*, c.hadm_id, c.primary_label, c.index_mi_time, c.hours_from_mi
            FROM ({base_select}) b
            JOIN cohort c
              ON b.subject_id = c.subject_id
             AND b.study_id = c.study_id
            ORDER BY b.subject_id, b.study_id, b.ecg_time
            """
        else:
            query = base_select + "\nORDER BY rl.subject_id, rl.study_id, rl.ecg_time"

        if limit is not None and limit > 0:
            query = query + f"\nLIMIT {int(limit)}"

        df = con.execute(query).fetchdf()
        return df.reset_index(drop=True)
    finally:
        con.close()


def deduplicate_mimic_index_by_study_id(index_df: pd.DataFrame, audit_csv: Optional[Path] = None) -> pd.DataFrame:
    if index_df.empty or "study_id" not in index_df.columns:
        return index_df

    duplicate_mask = index_df.duplicated(subset=["study_id"], keep=False)
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count == 0:
        return index_df.drop_duplicates(subset=["study_id"]).reset_index(drop=True)

    work_df = index_df.copy()
    work_df["_orig_order"] = np.arange(len(work_df), dtype=np.int64)

    if "hours_from_mi" in work_df.columns:
        work_df["_abs_hours_from_mi"] = pd.to_numeric(work_df["hours_from_mi"], errors="coerce").abs()
    else:
        work_df["_abs_hours_from_mi"] = np.nan

    if "index_mi_time" in work_df.columns:
        work_df["_index_mi_time_sort"] = pd.to_datetime(work_df["index_mi_time"], errors="coerce")
    else:
        work_df["_index_mi_time_sort"] = pd.NaT

    if "ecg_time" in work_df.columns:
        work_df["_ecg_time_sort"] = pd.to_datetime(work_df["ecg_time"], errors="coerce")
    else:
        work_df["_ecg_time_sort"] = pd.NaT

    ranked_df = work_df.sort_values(
        by=["study_id", "_abs_hours_from_mi", "_index_mi_time_sort", "_ecg_time_sort", "_orig_order"],
        ascending=[True, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked_df["_dedupe_rank"] = ranked_df.groupby("study_id", sort=False).cumcount() + 1
    ranked_df["_keep_row"] = ranked_df["_dedupe_rank"] == 1

    duplicate_ids = set(ranked_df.loc[ranked_df.duplicated(subset=["study_id"], keep=False), "study_id"].tolist())
    if audit_csv is not None and duplicate_ids:
        audit_df = ranked_df[ranked_df["study_id"].isin(duplicate_ids)].copy()
        audit_df = audit_df.sort_values(by=["study_id", "_dedupe_rank"]).reset_index(drop=True)
        audit_csv.parent.mkdir(parents=True, exist_ok=True)
        audit_df.to_csv(audit_csv, index=False)
        print(
            "Study ID dedupe audit saved: "
            f"{audit_csv} (rows={len(audit_df):,}, study_ids={len(duplicate_ids):,})"
        )

    deduped_df = ranked_df[ranked_df["_keep_row"]].copy()
    deduped_df = deduped_df.sort_values(by="_orig_order").reset_index(drop=True)
    deduped_df = deduped_df.drop(
        columns=[
            "_orig_order",
            "_abs_hours_from_mi",
            "_index_mi_time_sort",
            "_ecg_time_sort",
            "_dedupe_rank",
            "_keep_row",
        ],
        errors="ignore",
    )

    removed_rows = len(index_df) - len(deduped_df)
    print(
        "Resolved duplicate study_id rows: "
        f"removed {removed_rows:,} rows across {len(duplicate_ids):,} study_id values; "
        "kept one row per study_id using priority "
        "abs(hours_from_mi) -> index_mi_time -> ecg_time -> original order."
    )

    return deduped_df


def load_ecgdeli_features(
    source_csv: Optional[Path],
    source_parquet: Optional[Path],
    source_db: Optional[Path],
    source_table: Optional[str],
    join_key: str,
) -> pd.DataFrame:
    required_cols = [join_key] + ECGDELI_FEATURES

    if all(x is None for x in [source_csv, source_parquet, source_db]):
        print("⚠️  No ECGDeli source provided; ECGDeli features will be NaN")
        return pd.DataFrame(columns=required_cols)

    if source_csv is not None and source_csv.exists():
        ecgdeli_df = pd.read_csv(source_csv)
    elif source_parquet is not None and source_parquet.exists():
        ecgdeli_df = pd.read_parquet(source_parquet)
    elif source_db is not None and source_db.exists() and source_table:
        con = duckdb.connect(str(source_db))
        try:
            ecgdeli_df = con.execute(f"SELECT * FROM {source_table}").fetchdf()
        finally:
            con.close()
    else:
        raise FileNotFoundError(
            "No ECGDeli source found. Provide one of: --ecgdeli-csv, --ecgdeli-parquet, or --ecgdeli-db + --ecgdeli-table"
        )

    if join_key not in ecgdeli_df.columns:
        if join_key == "study_id" and "ecg_id" in ecgdeli_df.columns:
            ecgdeli_df = ecgdeli_df.rename(columns={"ecg_id": "study_id"})
        else:
            raise ValueError(f"Join key '{join_key}' not found in ECGDeli source columns")

    for col in ECGDELI_FEATURES:
        if col not in ecgdeli_df.columns:
            ecgdeli_df[col] = np.nan

    return ecgdeli_df[required_cols].drop_duplicates(subset=[join_key]).reset_index(drop=True)


def load_mimic_ecg_signal(waveform_path: str, mimic_ecg_dir: Path) -> Optional[tuple[np.ndarray, int]]:
    try:
        if waveform_path is None or pd.isna(waveform_path):
            return None

        clean_path = str(waveform_path).strip()
        if clean_path.endswith(".hea") or clean_path.endswith(".dat"):
            clean_path = clean_path[:-4]

        full_path = (mimic_ecg_dir / clean_path).as_posix()
        record = wfdb.rdrecord(full_path)
        signal = record.p_signal
        fs = getattr(record, "fs", None)

        if signal is None or not isinstance(signal, np.ndarray):
            return None
        if signal.ndim != 2 or signal.shape[1] < 12:
            return None
        if fs is None or float(fs) <= 0:
            return None

        return signal, int(fs)
    except Exception:
        return None


def extract_neurokit2_feature_row(ecg_signal: np.ndarray, sampling_rate: int) -> Dict[str, float]:
    row = {feature: np.nan for feature in NEUROKIT2_FEATURES}

    r_leads = ["V1", "V2", "aVR"]
    s_leads = ["I", "II", "III", "aVF", "aVL", "aVR", "V1", "V2", "V3", "V4"]
    q_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    st_leads = ["I", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]

    unique_leads: List[str] = []
    for lead in r_leads + s_leads + q_leads + st_leads:
        if lead not in unique_leads:
            unique_leads.append(lead)

    lead_contexts: Dict[str, Optional[Dict[str, object]]] = {
        lead: _build_neurokit_lead_context(ecg_signal, LEAD_MAP[lead], sampling_rate=sampling_rate)
        for lead in unique_leads
    }

    for lead in r_leads:
        result = calculate_r_amplitude_neurokit(
            ecg_signal,
            LEAD_MAP[lead],
            sampling_rate=sampling_rate,
            lead_context=lead_contexts.get(lead),
        )
        row[f"R_Amp_{lead}"] = result["value"]
        row[f"R_Amp_{lead}_count"] = result["count"]
        row[f"R_Amp_{lead}_iqr"] = result["iqr"]

    for lead in s_leads:
        result = calculate_s_amplitude_neurokit(
            ecg_signal,
            LEAD_MAP[lead],
            sampling_rate=sampling_rate,
            lead_context=lead_contexts.get(lead),
        )
        row[f"S_Amp_{lead}"] = result["value"]
        row[f"S_Amp_{lead}_count"] = result["count"]
        row[f"S_Amp_{lead}_iqr"] = result["iqr"]

    for lead in q_leads:
        result = calculate_q_amplitude_neurokit(
            ecg_signal,
            LEAD_MAP[lead],
            sampling_rate=sampling_rate,
            lead_context=lead_contexts.get(lead),
        )
        row[f"Q_Amp_{lead}"] = result["value"]
        row[f"Q_Amp_{lead}_count"] = result["count"]
        row[f"Q_Amp_{lead}_iqr"] = result["iqr"]

    for lead in st_leads:
        result = calculate_st_elevation_neurokit(
            ecg_signal,
            LEAD_MAP[lead],
            sampling_rate=sampling_rate,
            st_offset_ms=60,
            lead_context=lead_contexts.get(lead),
        )
        row[f"ST_Elev_{lead}"] = result["value"]
        row[f"ST_Elev_{lead}_count"] = result["count"]
        row[f"ST_Elev_{lead}_iqr"] = result["iqr"]

    return row


def _summarize_neurokit_status(nk_features: Dict[str, float]) -> tuple[str, int]:
    count_cols = [c for c in NEUROKIT2_FEATURES if c.endswith("_count")]
    total_detected = int(np.nansum([float(nk_features.get(c, 0) or 0) for c in count_cols]))
    if total_detected > 0:
        return "feature_extraction_completed", total_detected
    return "feature_extraction_empty", 0


def _extract_single_record(record: Dict[str, object], mimic_ecg_dir: str) -> Dict[str, object]:
    waveform_path = record.get("waveform_path")
    base = {
        "subject_id": record.get("subject_id", np.nan),
        "study_id": record.get("study_id", np.nan),
        "ecg_time": record.get("ecg_time", pd.NaT),
        "waveform_path": waveform_path,
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "sampling_rate": np.nan,
        "nk_total_detected_beats": 0,
        "extraction_status": "signal_load_failed",
    }

    if "hadm_id" in record:
        base["hadm_id"] = record.get("hadm_id")
    if "primary_label" in record:
        base["primary_label"] = record.get("primary_label")
    if "index_mi_time" in record:
        base["index_mi_time"] = record.get("index_mi_time")
    if "hours_from_mi" in record:
        base["hours_from_mi"] = record.get("hours_from_mi")

    loaded = load_mimic_ecg_signal(waveform_path=waveform_path, mimic_ecg_dir=Path(mimic_ecg_dir))
    if loaded is None:
        base.update({f: np.nan for f in NEUROKIT2_FEATURES})
        return base

    ecg_signal, sampling_rate = loaded
    nk_features = extract_neurokit2_feature_row(ecg_signal, sampling_rate=sampling_rate)
    status, total_detected = _summarize_neurokit_status(nk_features)

    base["sampling_rate"] = sampling_rate
    base["nk_total_detected_beats"] = total_detected
    base["extraction_status"] = status
    base.update(nk_features)
    return base


def _save_checkpoint(
    rows: List[Dict[str, object]],
    checkpoint_dir: Path,
    processed_count: int,
    last_checkpoint_row_idx: int,
) -> int:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    end_row = len(rows)
    start_row = max(0, last_checkpoint_row_idx)
    if end_row <= start_row:
        return last_checkpoint_row_idx

    new_rows = rows[start_row:end_row]
    delta_count = len(new_rows)
    global_end = int(processed_count)
    global_start = global_end - delta_count + 1
    checkpoint_path = checkpoint_dir / f"hybrid_192_delta_{global_start}_{global_end}.parquet"
    pd.DataFrame(new_rows).to_parquet(checkpoint_path, index=False)
    print(
        f"\n✓ Checkpoint saved: {checkpoint_path} "
        f"(rows={processed_count:,}, delta={len(new_rows):,})"
    )
    return end_row


def _get_latest_checkpoint_path(checkpoint_dir: Path) -> Optional[Path]:
    if not checkpoint_dir.exists():
        return None

    rolling_path = checkpoint_dir / ROLLING_CHECKPOINT
    if rolling_path.exists():
        return rolling_path

    pattern = re.compile(r"^hybrid_192_checkpoint_(\d+)\.parquet$")
    latest_path: Optional[Path] = None
    latest_count = -1

    for candidate in checkpoint_dir.glob("hybrid_192_checkpoint_*.parquet"):
        match = pattern.match(candidate.name)
        if not match:
            continue
        count = int(match.group(1))
        if count > latest_count:
            latest_count = count
            latest_path = candidate

    return latest_path


def _load_latest_checkpoint_rows(checkpoint_dir: Path) -> List[Dict[str, object]]:
    delta_files: List[tuple[int, int, Path]] = []
    if checkpoint_dir.exists():
        for candidate in checkpoint_dir.glob("hybrid_192_delta_*.parquet"):
            match = DELTA_CHECKPOINT_PATTERN.match(candidate.name)
            if not match:
                continue
            start_idx = int(match.group(1))
            end_idx = int(match.group(2))
            delta_files.append((start_idx, end_idx, candidate))

    if delta_files:
        delta_files.sort(key=lambda item: (item[0], item[1]))
        starts_at_one = delta_files[0][0] == 1
        contiguous = True
        for i in range(1, len(delta_files)):
            prev_end = delta_files[i - 1][1]
            curr_start = delta_files[i][0]
            if curr_start != prev_end + 1:
                contiguous = False
                break

        if starts_at_one and contiguous:
            chunks = [pd.read_parquet(path) for _, _, path in delta_files]
            checkpoint_df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            rows = checkpoint_df.to_dict(orient="records")
            print(
                f"Resuming from delta checkpoints: {len(delta_files)} file(s) "
                f"(rows={len(rows):,})"
            )
            return rows

    latest_path = _get_latest_checkpoint_path(checkpoint_dir)
    if latest_path is None:
        if delta_files:
            first_start = delta_files[0][0]
            raise ValueError(
                "Delta checkpoint continuity error: found delta files but no base checkpoint and first "
                f"delta starts at row {first_start} instead of 1."
            )
        return []

    checkpoint_df = pd.read_parquet(latest_path)
    base_rows = checkpoint_df.to_dict(orient="records")

    if not delta_files:
        print(f"Resuming from checkpoint: {latest_path} (rows={len(base_rows):,})")
        return base_rows

    expected_next = len(base_rows) + 1
    tail_deltas = [(s, e, p) for s, e, p in delta_files if s >= expected_next]

    if not tail_deltas:
        print(
            f"Resuming from checkpoint: {latest_path} (rows={len(base_rows):,}); "
            "delta files detected but none extend beyond base checkpoint."
        )
        return base_rows

    if tail_deltas[0][0] != expected_next:
        raise ValueError(
            "Delta checkpoint gap detected after base checkpoint: "
            f"base rows={len(base_rows):,}, next delta starts at {tail_deltas[0][0]:,}."
        )

    for i in range(1, len(tail_deltas)):
        prev_end = tail_deltas[i - 1][1]
        curr_start = tail_deltas[i][0]
        if curr_start != prev_end + 1:
            raise ValueError(
                "Delta checkpoint gap detected: "
                f"delta ends at {prev_end:,}, next starts at {curr_start:,}."
            )

    delta_chunks = [pd.read_parquet(path) for _, _, path in tail_deltas]
    if delta_chunks:
        combined_df = pd.concat([checkpoint_df] + delta_chunks, ignore_index=True)
    else:
        combined_df = checkpoint_df

    rows = combined_df.to_dict(orient="records")
    print(
        f"Resuming from checkpoint+delta: base={latest_path.name}, "
        f"delta_files={len(tail_deltas)}, rows={len(rows):,}"
    )
    return rows


def _escape_for_matlab_string(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def run_matlab_ecgdeli(
    matlab_executable: str,
    matlab_batch_command: Optional[str],
    matlab_script: Optional[Path],
    index_csv: Path,
    output_csv: Path,
    matlab_workers: int,
    start_row: int,
    end_row: int,
    append_output: bool,
    timeout_seconds: int,
    progress_miniters: int = 10,
    env_overrides: Optional[Dict[str, str]] = None,
) -> None:
    if matlab_batch_command:
        command_text = matlab_batch_command.format(
            index_csv=_escape_for_matlab_string(index_csv),
            output_csv=_escape_for_matlab_string(output_csv),
            start_row=int(start_row),
            end_row=int(end_row),
            append_output=int(bool(append_output)),
        )
    else:
        if matlab_script is None:
            raise ValueError("Provide --matlab-batch-command or --matlab-script when using --run-matlab-ecgdeli")
        if not matlab_script.exists():
            raise FileNotFoundError(f"MATLAB script not found: {matlab_script}")

        script_dir = _escape_for_matlab_string(matlab_script.parent)
        script_name = matlab_script.stem
        index_csv_escaped = _escape_for_matlab_string(index_csv)
        output_csv_escaped = _escape_for_matlab_string(output_csv)
        workers = max(1, int(matlab_workers))
        matlab_start_row = max(1, int(start_row or 1))
        matlab_end_row = int(end_row or 0)
        matlab_append_flag = 1 if append_output else 0
        command_text = (
            f"addpath('{script_dir}'); "
            f"{script_name}('{index_csv_escaped}','{output_csv_escaped}',{workers},{matlab_start_row},{matlab_end_row},{matlab_append_flag});"
        )

    cmd = [matlab_executable, "-batch", command_text]
    timeout = None if timeout_seconds <= 0 else timeout_seconds
    run_env = None
    if env_overrides:
        run_env = dict(**os.environ)
        run_env.update(env_overrides)

    print("\nRunning MATLAB ECGDeli extraction...")
    print(f"MATLAB executable: {matlab_executable}")
    print(f"MATLAB workers: {max(1, int(matlab_workers))}")
    print(f"MATLAB row range: {int(start_row)}..{int(end_row)}")
    print(f"MATLAB append output: {bool(append_output)}")
    print(f"MATLAB output CSV target: {output_csv}")

    expected_records: Optional[int] = None
    if int(start_row) > 0 and int(end_row) >= int(start_row):
        expected_records = int(end_row) - int(start_row) + 1
    elif index_csv.exists():
        try:
            expected_records = max(0, len(pd.read_csv(index_csv)))
        except Exception:
            expected_records = None

    start_time = time.time()
    matlab_log_path = output_csv.parent / "matlab_ecgdeli_run.log"
    matlab_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(matlab_log_path, "w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, text=True, env=run_env)

        pbar = None
        interactive_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        try:
            pbar = tqdm(
                total=expected_records,
                desc="MATLAB ECGDeli (records)",
                unit="rec",
                dynamic_ncols=True,
                disable=not interactive_tty,
                mininterval=2.0,
                miniters=max(1, progress_miniters),
            )
        except TypeError:
            pbar = None

        log_read_pos = 0
        processed_records = 0
        last_console_print_sec = -1
        progress_pattern = re.compile(r"\[(\d+)/(\d+)\]")

        while process.poll() is None:
            elapsed = int(time.time() - start_time)

            output_size_mb = 0.0
            if output_csv.exists():
                output_size_mb = output_csv.stat().st_size / (1024 * 1024)

            latest_total_from_log: Optional[int] = None
            latest_processed_from_log: Optional[int] = None
            try:
                with open(matlab_log_path, "r", encoding="utf-8", errors="replace") as lf:
                    lf.seek(log_read_pos)
                    chunk = lf.read()
                    log_read_pos = lf.tell()
                if chunk:
                    for match in progress_pattern.finditer(chunk):
                        latest_processed_from_log = int(match.group(1))
                        latest_total_from_log = int(match.group(2))
            except Exception:
                pass

            if latest_processed_from_log is not None:
                if latest_total_from_log and pbar is not None and pbar.total is None:
                    pbar.total = latest_total_from_log
                if latest_total_from_log:
                    expected_records = latest_total_from_log
                processed_records = max(processed_records, latest_processed_from_log)

            if pbar is not None:
                if processed_records > pbar.n:
                    pbar.update(processed_records - pbar.n)
                pbar.set_postfix(
                    output_csv_mb=f"{output_size_mb:.2f}",
                    timeout_s=(str(timeout) if timeout is not None else "none"),
                    refresh=False,
                )
            elif elapsed % 15 == 0 and elapsed > 0 and elapsed != last_console_print_sec:
                total_text = str(expected_records) if expected_records is not None else "?"
                print(
                    "MATLAB ECGDeli running... "
                    f"elapsed={elapsed}s, processed={processed_records}/{total_text}, "
                    f"output_csv_mb={output_size_mb:.2f}"
                )
                last_console_print_sec = elapsed

            if timeout is not None and elapsed >= timeout:
                process.kill()
                process.wait(timeout=30)
                if pbar is not None:
                    pbar.close()
                details = ""
                try:
                    tail_lines = matlab_log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
                    details = "\n".join(tail_lines).strip()
                except Exception:
                    details = ""
                raise TimeoutError(
                    "MATLAB ECGDeli extraction timed out after "
                    f"{timeout} seconds. See log: {matlab_log_path}. "
                    f"Last lines:\n{details}"
                )

            time.sleep(1)

        if pbar is not None:
            if expected_records is not None and expected_records > pbar.n:
                pbar.update(expected_records - pbar.n)
            pbar.close()

    returncode = process.returncode or 0
    if returncode != 0:
        details = ""
        try:
            tail_lines = matlab_log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            details = "\n".join(tail_lines).strip()
        except Exception:
            details = ""
        raise RuntimeError(
            "MATLAB ECGDeli extraction failed "
            f"(exit code {returncode}). See log: {matlab_log_path}. "
            f"Last lines:\n{details}"
        )

    if not output_csv.exists():
        raise FileNotFoundError(
            "MATLAB run completed but ECGDeli output CSV was not found at "
            f"{output_csv}. Ensure your MATLAB script writes this file."
        )

    print(f"✓ MATLAB ECGDeli extraction completed: {output_csv}")
    print(f"MATLAB log: {matlab_log_path}")


def _resolve_matlab_workers(requested_workers: int) -> int:
    if requested_workers > 0:
        return max(1, int(requested_workers))

    env_workers = os.environ.get("MATLAB_ECGDELI_WORKERS", "").strip()
    if env_workers:
        try:
            parsed = int(env_workers)
            if parsed > 0:
                return parsed
        except ValueError:
            pass

    return max(1, os.cpu_count() or 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid MIMIC-IV ECG feature extraction (192 features)")
    parser.add_argument("--db-path", type=Path, default=Path("data/mimic_database.duckdb"))
    parser.add_argument("--mimic-ecg-dir", type=Path, default=Path("data/raw/MIMIC-IV-ECG-1.0"))
    parser.add_argument("--cohort-path", type=Path, default=Path("data/processed/cohort_master.parquet"))
    parser.add_argument(
        "--canonical-index-path",
        type=Path,
        default=None,
        help=(
            "Path for persisted canonical deduped index used for stable resume ordering across runs. "
            "Default: <output folder>/canonical_index.parquet"
        ),
    )
    parser.add_argument(
        "--rebuild-canonical-index",
        action="store_true",
        help="Rebuild canonical deduped index from source tables even if canonical index file already exists.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--join-key", type=str, default="study_id", choices=["study_id"])

    parser.add_argument("--ecgdeli-csv", type=Path, default=None)
    parser.add_argument("--ecgdeli-parquet", type=Path, default=None)
    parser.add_argument("--ecgdeli-db", type=Path, default=None)
    parser.add_argument("--ecgdeli-table", type=str, default=None)
    parser.add_argument("--require-ecgdeli", action="store_true")

    parser.add_argument("--run-matlab-ecgdeli", action="store_true")
    parser.add_argument("--matlab-executable", type=str, default="matlab")
    parser.add_argument("--matlab-batch-command", type=str, default=None)
    parser.add_argument("--matlab-script", type=Path, default=Path("scripts/matlab/extract_mimic_ecgdeli_features.m"))
    parser.add_argument("--matlab-index-csv", type=Path, default=Path("data/interim/mimic_index_for_ecgdeli.csv"))
    parser.add_argument("--matlab-output-csv", type=Path, default=Path("data/processed/mimic_ecgdeli_features.csv"))
    parser.add_argument("--matlab-timeout-seconds", type=int, default=0)
    parser.add_argument("--matlab-workers", type=int, default=0, help="MATLAB ECGDeli worker count (0=auto/use all logical CPUs)")
    parser.add_argument("--matlab-checkpoint-every", type=int, default=1000, help="MATLAB ECGDeli progress/checkpoint interval in records (lower = more live updates)")
    parser.add_argument("--matlab-ecgdeli-path", type=Path, default=None)
    parser.add_argument("--matlab-wfdb-path", type=Path, default=None)
    parser.add_argument("--matlab-mimic-ecg-root", type=Path, default=Path("data/raw/MIMIC-IV-ECG-1.0"))

    parser.add_argument("--parallel-processes", type=int, default=1)
    parser.add_argument(
        "--gpu-stats",
        action="store_true",
        help=(
            "Accelerate summary statistics (mean/count/iqr) using GPU (PyTorch/CUDA). "
            f"GPU path is used only when beat-array size >= {MIN_GPU_STATS_SIZE}. "
            "Not recommended with --parallel-processes > 1 due to single-GPU contention."
        ),
    )
    parser.add_argument("--gpu-device", type=str, default="cuda:0", help="CUDA device for --gpu-stats, e.g. cuda:0")
    parser.add_argument("--gpu-allow-cpu-fallback", action="store_true", help="Allow CPU fallback on CUDA OOM (default is strict fail)")
    parser.add_argument("--gpu-oom-probe", action="store_true", help="Intentionally trigger CUDA OOM preflight to produce evidence/logs")
    parser.add_argument("--gpu-reserve-mb", type=int, default=0, help="Persistently reserve this many MB of VRAM per Python process when --gpu-stats is enabled")
    parser.add_argument("--gpu-reserve-fraction", type=float, default=0.0, help="Persistently reserve this fraction (0-1) of currently free VRAM per Python process when --gpu-stats is enabled")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume-from-latest-checkpoint", action="store_true", help="Resume extraction from latest checkpoint in --checkpoint-dir (or default checkpoint folder)")
    parser.add_argument("--max-records-per-run", type=int, default=0, help="Maximum number of records to process in this run after resume offset (0 = no cap)")
    parser.add_argument("--tqdm-miniters", type=int, default=10, help="Progress bar update frequency in records (default: 10)")
    parser.add_argument("--suppress-neurokit-warnings", dest="suppress_neurokit_warnings", action="store_true", help="Suppress frequent NeuroKit runtime warnings for cleaner logs")
    parser.add_argument("--show-neurokit-warnings", dest="suppress_neurokit_warnings", action="store_false", help="Show NeuroKit runtime warnings")
    parser.set_defaults(suppress_neurokit_warnings=True)

    parser.add_argument("--output-parquet", type=Path, default=Path("data/processed/mimic_hybrid_192_features.parquet"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/processed/mimic_hybrid_192_features.csv"))
    parser.add_argument(
        "--study-id-duplicate-audit-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV path to save all duplicate study_id candidates and their dedupe rank. "
            "Default: <output folder>/cohort_study_id_duplicate_audit.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_matlab_workers = _resolve_matlab_workers(args.matlab_workers)

    if args.suppress_neurokit_warnings:
        warnings.filterwarnings(
            "ignore",
            message=r"Too few peaks detected to compute the rate\. Returning empty vector\.",
            module=r"neurokit2\.signal\.signal_period",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"There are .* missing data points in your signal\. Filling missing values by using the forward filling method\.",
            module=r"neurokit2\.ecg\.ecg_clean",
        )

    assert len(NEUROKIT2_FEATURES) == 99, f"Expected 99 NeuroKit2 features, got {len(NEUROKIT2_FEATURES)}"
    assert len(ECGDELI_FEATURES) == 93, f"Expected 93 ECGDeli features, got {len(ECGDELI_FEATURES)}"

    print("=" * 80)
    print("HYBRID MIMIC-IV ECG FEATURE EXTRACTION")
    print("=" * 80)
    print(f"NeuroKit2 features: {len(NEUROKIT2_FEATURES)}")
    print(f"ECGDeli features: {len(ECGDELI_FEATURES)}")
    print(f"Total target features: {len(NEUROKIT2_FEATURES) + len(ECGDELI_FEATURES)}")
    print(f"Parallel processes: {max(1, args.parallel_processes)}")
    if args.gpu_stats and max(1, args.parallel_processes) > 1:
        print(
            "⚠️  WARNING: --gpu-stats with --parallel-processes > 1 may reduce performance "
            "or increase GPU memory pressure due to multi-process contention on one GPU."
        )

    if args.gpu_stats and _torch_cuda_available():
        torch_module = _get_torch_module()
        if torch_module is None:
            raise RuntimeError("--gpu-stats requested, but PyTorch is not available in this environment.")
        os.environ["HYBRID_GPU_STATS"] = "1"
        os.environ["HYBRID_GPU_DEVICE"] = args.gpu_device
        os.environ["HYBRID_GPU_STRICT_OOM"] = "0" if args.gpu_allow_cpu_fallback else "1"
        os.environ["HYBRID_GPU_RESERVE_MB"] = str(max(0, args.gpu_reserve_mb))
        os.environ["HYBRID_GPU_RESERVE_FRACTION"] = str(max(0.0, min(1.0, args.gpu_reserve_fraction)))
        gpu_device = torch_module.device(args.gpu_device)
        gpu_name = torch_module.cuda.get_device_name(gpu_device) if args.gpu_device.startswith("cuda") else "CUDA"
        free_bytes, total_bytes = torch_module.cuda.mem_get_info(gpu_device)
        print(f"GPU stats acceleration: enabled ({args.gpu_device} | {gpu_name})")
        print(f"GPU OOM behavior: {'CPU fallback' if args.gpu_allow_cpu_fallback else 'strict fail'}")
        print(
            "GPU memory: "
            f"{free_bytes / 1024**2:.1f} MB free / {total_bytes / 1024**2:.1f} MB total"
        )
        print(
            f"GPU reservation request: {max(0, args.gpu_reserve_mb)} MB, "
            f"fraction={max(0.0, min(1.0, args.gpu_reserve_fraction))}"
        )

        if args.gpu_oom_probe:
            _run_gpu_oom_probe(args.gpu_device)
    else:
        os.environ["HYBRID_GPU_STATS"] = "0"
        os.environ["HYBRID_GPU_STRICT_OOM"] = "1"
        os.environ["HYBRID_GPU_RESERVE_MB"] = "0"
        os.environ["HYBRID_GPU_RESERVE_FRACTION"] = "0"
        if args.gpu_stats:
            if args.gpu_allow_cpu_fallback:
                print("GPU stats acceleration: requested but CUDA/PyTorch not available; using CPU fallback (--gpu-allow-cpu-fallback)")
            else:
                raise RuntimeError(
                    "--gpu-stats requested, but CUDA/PyTorch is not available in this environment. "
                    "Use --gpu-allow-cpu-fallback to continue on CPU."
                )
        else:
            print("GPU stats acceleration: disabled")
    print(f"Checkpoint frequency: {args.checkpoint_every} rows" if args.checkpoint_every > 0 else "Checkpoint frequency: disabled")
    print(f"Resume from latest checkpoint: {args.resume_from_latest_checkpoint}")
    print(f"Max records per run: {args.max_records_per_run if args.max_records_per_run > 0 else 'all remaining'}")
    print(f"Suppress NeuroKit warnings: {args.suppress_neurokit_warnings}")
    print(f"Require ECGDeli completeness: {args.require_ecgdeli}")
    print(f"Run MATLAB ECGDeli: {args.run_matlab_ecgdeli}")
    if args.run_matlab_ecgdeli:
        print(f"MATLAB ECGDeli workers: {resolved_matlab_workers}")
        print(f"MATLAB checkpoint/progress interval: {max(1, args.matlab_checkpoint_every)}")
    print()

    canonical_index_path = args.canonical_index_path
    if canonical_index_path is None:
        canonical_index_path = args.output_csv.parent / "canonical_index.parquet"

    if canonical_index_path.exists() and not args.rebuild_canonical_index:
        index_df = pd.read_parquet(canonical_index_path)
        print(f"Loaded canonical deduped index: {canonical_index_path} (rows={len(index_df):,})")
    else:
        index_df = load_mimic_record_index(
            db_path=args.db_path,
            cohort_path=args.cohort_path,
            limit=args.limit,
        )
        print(f"Loaded MIMIC index rows: {len(index_df):,}")

        study_id_audit_csv = args.study_id_duplicate_audit_csv
        if study_id_audit_csv is None:
            study_id_audit_csv = args.output_csv.parent / "cohort_study_id_duplicate_audit.csv"
        index_df = deduplicate_mimic_index_by_study_id(index_df, audit_csv=study_id_audit_csv)
        print(f"Index rows after study_id dedupe: {len(index_df):,}")

        canonical_index_path.parent.mkdir(parents=True, exist_ok=True)
        index_df.to_parquet(canonical_index_path, index=False)
        print(f"Saved canonical deduped index: {canonical_index_path}")

    checkpoint_dir = args.checkpoint_dir or (args.output_parquet.parent / "checkpoints_hybrid_192")
    rows: List[Dict[str, object]] = []
    if args.resume_from_latest_checkpoint:
        rows = _load_latest_checkpoint_rows(checkpoint_dir)

    records = index_df.to_dict(orient="records")
    initial_completed_rows = len(rows)
    total_target_records = len(records)
    if initial_completed_rows > 0:
        if initial_completed_rows >= total_target_records:
            print(
                "Checkpoint already covers all indexed records "
                f"({initial_completed_rows:,}/{total_target_records:,}); skipping NeuroKit extraction loop."
            )
            records = []
        else:
            records = records[initial_completed_rows:]
            print(
                "Resume applied: "
                f"skipping first {initial_completed_rows:,} records; "
                f"remaining {len(records):,} records."
            )

    max_records_per_run = max(0, int(args.max_records_per_run or 0))
    if max_records_per_run > 0 and len(records) > max_records_per_run:
        records = records[:max_records_per_run]
        print(
            "Per-run cap applied: "
            f"processing {len(records):,} records this run "
            f"(remaining overall after this run: {total_target_records - (initial_completed_rows + len(records)):,})."
        )

    planned_run_count = len(records)
    matlab_start_row = initial_completed_rows + 1 if planned_run_count > 0 else 0
    matlab_end_row = initial_completed_rows + planned_run_count if planned_run_count > 0 else 0
    matlab_append_output = bool(args.resume_from_latest_checkpoint and initial_completed_rows > 0)

    if args.run_matlab_ecgdeli:
        args.matlab_index_csv.parent.mkdir(parents=True, exist_ok=True)
        args.matlab_output_csv.parent.mkdir(parents=True, exist_ok=True)
        index_export_cols = [c for c in ["subject_id", "study_id", "ecg_time", "waveform_path", "hadm_id", "primary_label"] if c in index_df.columns]
        index_df[index_export_cols].to_csv(args.matlab_index_csv, index=False)
        print(f"Exported MATLAB ECGDeli index: {args.matlab_index_csv}")

        if planned_run_count <= 0:
            print("Skipping MATLAB ECGDeli: no records planned for this resumed run.")
        else:
            run_matlab_ecgdeli(
                matlab_executable=args.matlab_executable,
                matlab_batch_command=args.matlab_batch_command,
                matlab_script=args.matlab_script,
                index_csv=args.matlab_index_csv,
                output_csv=args.matlab_output_csv,
                matlab_workers=resolved_matlab_workers,
                start_row=matlab_start_row,
                end_row=matlab_end_row,
                append_output=matlab_append_output,
                timeout_seconds=args.matlab_timeout_seconds,
                progress_miniters=args.tqdm_miniters,
                env_overrides={
                    "ECGDELI_PATH": str(args.matlab_ecgdeli_path) if args.matlab_ecgdeli_path else "",
                    "WFDB_PATH": str(args.matlab_wfdb_path) if args.matlab_wfdb_path else "",
                    "MIMIC_ECG_ROOT": str(args.matlab_mimic_ecg_root) if args.matlab_mimic_ecg_root else "",
                    "MATLAB_ECGDELI_WORKERS": str(resolved_matlab_workers),
                    "MATLAB_ECGDELI_CHECKPOINT_EVERY": str(max(1, args.matlab_checkpoint_every)),
                },
            )

    effective_ecgdeli_csv = args.ecgdeli_csv
    if args.run_matlab_ecgdeli and effective_ecgdeli_csv is None:
        effective_ecgdeli_csv = args.matlab_output_csv
        print(f"Using MATLAB ECGDeli output as source CSV: {effective_ecgdeli_csv}")

    ecgdeli_df = load_ecgdeli_features(
        source_csv=effective_ecgdeli_csv,
        source_parquet=args.ecgdeli_parquet,
        source_db=args.ecgdeli_db,
        source_table=args.ecgdeli_table,
        join_key=args.join_key,
    )

    if args.require_ecgdeli and ecgdeli_df.empty:
        raise ValueError(
            "--require-ecgdeli is enabled, but no ECGDeli source rows were loaded. "
            "Provide a valid MIMIC-IV ECGDeli source via --ecgdeli-csv/--ecgdeli-parquet/--ecgdeli-db."
        )

    print(f"Loaded ECGDeli rows: {len(ecgdeli_df):,}")

    interactive_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    last_checkpoint_row_idx = len(rows)

    if max(1, args.parallel_processes) > 1:
        worker = partial(_extract_single_record, mimic_ecg_dir=str(args.mimic_ecg_dir))
        with ProcessPoolExecutor(max_workers=max(1, args.parallel_processes)) as executor:
            iterator = executor.map(worker, records)
            progress_iter = tqdm(
                iterator,
                total=len(records),
                desc="Extracting NeuroKit2 features (parallel)",
                mininterval=0.5,
                miniters=max(1, args.tqdm_miniters),
                dynamic_ncols=True,
                disable=not interactive_tty,
            )
            for idx, result in enumerate(progress_iter, start=1):
                rows.append(result)
                processed_total = initial_completed_rows + idx
                if args.checkpoint_every > 0 and processed_total % args.checkpoint_every == 0:
                    last_checkpoint_row_idx = _save_checkpoint(rows, checkpoint_dir, processed_total, last_checkpoint_row_idx)
                if not interactive_tty and idx % max(1, args.tqdm_miniters) == 0:
                    print(f"Extracting NeuroKit2 features (parallel): {processed_total}/{total_target_records}")
    else:
        progress_iter = tqdm(
            records,
            total=len(records),
            desc="Extracting NeuroKit2 features",
            mininterval=0.5,
            miniters=max(1, args.tqdm_miniters),
            dynamic_ncols=True,
            disable=not interactive_tty,
        )
        for idx, record in enumerate(progress_iter, start=1):
            rows.append(_extract_single_record(record, str(args.mimic_ecg_dir)))
            processed_total = initial_completed_rows + idx
            if args.checkpoint_every > 0 and processed_total % args.checkpoint_every == 0:
                last_checkpoint_row_idx = _save_checkpoint(rows, checkpoint_dir, processed_total, last_checkpoint_row_idx)
            if not interactive_tty and idx % max(1, args.tqdm_miniters) == 0:
                print(f"Extracting NeuroKit2 features: {processed_total}/{total_target_records}")

    neurokit_df = pd.DataFrame(rows)

    merged = neurokit_df.merge(ecgdeli_df, on=args.join_key, how="left")

    if args.require_ecgdeli:
        ecgdeli_nan_counts = merged[ECGDELI_FEATURES].isna().sum()
        rows_with_any_missing = int(merged[ECGDELI_FEATURES].isna().any(axis=1).sum())
        if rows_with_any_missing > 0:
            missing_examples = ecgdeli_nan_counts[ecgdeli_nan_counts > 0].sort_values(ascending=False).head(10)
            missing_summary = ", ".join([f"{col}={int(cnt)}" for col, cnt in missing_examples.items()])
            raise ValueError(
                "--require-ecgdeli is enabled, but final merged output contains missing ECGDeli values. "
                f"Rows affected: {rows_with_any_missing}/{len(merged)}. "
                f"Top missing columns: {missing_summary}"
            )

    full_feature_list = NEUROKIT2_FEATURES + ECGDELI_FEATURES
    missing_cols = [c for c in full_feature_list if c not in merged.columns]
    for col in missing_cols:
        merged[col] = np.nan

    final_columns = [
        c for c in [
            "subject_id",
            "study_id",
            "hadm_id",
            "ecg_time",
            "index_mi_time",
            "hours_from_mi",
            "primary_label",
            "waveform_path",
            "extraction_timestamp",
            "sampling_rate",
            "nk_total_detected_beats",
            "extraction_status",
        ] if c in merged.columns
    ] + full_feature_list
    final_df = merged[final_columns]

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    final_df.to_parquet(args.output_parquet, index=False)
    final_df.to_csv(args.output_csv, index=False)

    print()
    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Output rows: {len(final_df):,}")
    print(f"Output columns: {len(final_df.columns)}")
    print(f"Feature columns present: {sum([1 for c in full_feature_list if c in final_df.columns])}")
    print(f"NeuroKit2 feature count: {len(NEUROKIT2_FEATURES)}")
    print(f"ECGDeli feature count: {len(ECGDELI_FEATURES)}")
    print(f"Total expected features: {len(full_feature_list)}")
    print(f"Saved Parquet: {args.output_parquet}")
    print(f"Saved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
