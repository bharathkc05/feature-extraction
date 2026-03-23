# College PC Run Kit (MIMIC-IV Hybrid 192 Features)

This folder contains everything needed to run the hybrid extraction on a new PC with minimal failure risk.

All commands are written to be launched from the project root (one level above `deployment/`).

## 1) Required folder/file placement (must match)
Keep this layout under your project root:

- `scripts/phase_d_hybrid_192_feature_extraction.py`
- `scripts/matlab/extract_mimic_ecgdeli_features.m`
- `data/mimic_database.duckdb`
- `data/raw/MIMIC-IV-ECG-1.0/` (full waveform tree under `files/...`)
- `data/processed/` (output/checkpoints)
- `data/interim/` (MATLAB index export)

Optional but recommended:
- `data/processed/cohort_master.parquet` (if cohort-limited extraction is needed)

## 2) MATLAB prerequisites (required for `--run-matlab-ecgdeli`)
Inside MATLAB path, make sure these are available:
- ECGDeli toolbox (`Annotate_ECG_Multi`, `ExtractAmplitudeFeaturesFromFPT`, etc.)
- WFDB MATLAB functions (`rdsamp`)

Bundle behavior:
- `third_party/wfdb` is bundled in this package for portability.
- `setup_matlab_toolboxes.ps1` uses bundled WFDB first and downloads from PhysioNet only if local WFDB files are missing.

The Python pipeline passes environment variables to MATLAB when needed:
- `ECGDELI_PATH`
- `WFDB_PATH`
- `MIMIC_ECG_ROOT`

## 3) Environment setup
From project root:

### GPU setup (recommended)
```powershell
powershell -ExecutionPolicy Bypass -File deployment/college_pc_runkit/setup_college_env.ps1 -Gpu
```

### CPU setup (fallback)
```powershell
powershell -ExecutionPolicy Bypass -File deployment/college_pc_runkit/setup_college_env.ps1
```

Note: both GPU and CPU setup create and use `.\cuda_env\Scripts\python.exe`.

## 3.5) Optional MATLAB worker auto-tuning
If this machine has a strong CPU, you can auto-increase MATLAB `Processes` workers:

```powershell
powershell -ExecutionPolicy Bypass -File deployment/college_pc_runkit/tune_matlab_workers.ps1
```

Optional manual override:

```powershell
powershell -ExecutionPolicy Bypass -File deployment/college_pc_runkit/tune_matlab_workers.ps1 -Workers 16
```

Optional profile/log overrides:

```powershell
powershell -ExecutionPolicy Bypass -File deployment/college_pc_runkit/tune_matlab_workers.ps1 -ClusterProfile Processes -LogPath deployment/college_pc_runkit/logs/matlab_worker_tuning.log
```

## 4) Preflight validation
```powershell
.\cuda_env\Scripts\python.exe deployment/college_pc_runkit/verify_setup.py
.\cuda_env\Scripts\python.exe deployment/college_pc_runkit/validate_data_placement.py
.\cuda_env\Scripts\python.exe deployment/college_pc_runkit/check_wfdb_integrity.py
```

If your waveform dataset is stored in a non-default location, pass it explicitly:

```powershell
.\cuda_env\Scripts\python.exe deployment/college_pc_runkit/check_wfdb_integrity.py --path data/raw/MIMIC-IV-ECG-1.0
```

## 5) Run 100-sample extraction
```powershell
powershell -ExecutionPolicy Bypass -File deployment/college_pc_runkit/run_hybrid_100.ps1
```

By default, this run script first calls `tune_matlab_workers.ps1` to auto-tune MATLAB `Processes` workers.
It then exports the tuned value to `MATLAB_ECGDELI_WORKERS` for the Python pipeline.
If you want to skip that step:

```powershell
powershell -ExecutionPolicy Bypass -File deployment/college_pc_runkit/run_hybrid_100.ps1 -SkipMatlabWorkerTuning
```

Important: `run_hybrid_100.ps1` uses `--gpu-stats` in strict mode. On machines without CUDA, use this CPU-safe fallback command instead:

```powershell
powershell -ExecutionPolicy Bypass -File deployment/college_pc_runkit/run_hybrid_100_cpu.ps1
```

`run_hybrid_100_cpu.ps1` also auto-runs MATLAB worker tuning and exports `MATLAB_ECGDELI_WORKERS` by default.
To skip this in CPU mode:

```powershell
powershell -ExecutionPolicy Bypass -File deployment/college_pc_runkit/run_hybrid_100_cpu.ps1 -SkipMatlabWorkerTuning
```

## 6) Expected outputs
- `data/processed/test100_hybrid_192.csv`
- `data/processed/test100_hybrid_192.parquet`
- checkpoints in `data/processed/checkpoints_hybrid_192/`
- MATLAB log at `data/processed/matlab_ecgdeli_run.log`

## 7) Notes for reliability
- Use `--parallel-processes 1` when `--gpu-stats` is enabled (avoids single-GPU contention).
- If CUDA is unavailable and you still pass `--gpu-stats`, pipeline fails by design (strict mode).
- If you need CPU fallback behavior, add `--gpu-allow-cpu-fallback`.

## 8) Common failure points
- Missing waveform files under `data/raw/MIMIC-IV-ECG-1.0/files/...`
- MATLAB not in PATH (or missing required toolboxes)
- Wrong Python environment (run with `cuda_env\Scripts\python.exe` only)
- Missing write permissions in `data/processed`
- Running from the wrong working directory (always run commands from project root)
