# College PC Bundle (Zip-Ready, No Data)

This folder is prepared to be zipped and moved to another PC.
It intentionally excludes real data files.

All commands in this README use relative paths, so the extracted folder can be in any location on the target machine.

## What is included
- `scripts/phase_d_hybrid_192_feature_extraction.py`
- `scripts/matlab/extract_mimic_ecgdeli_features.m`
- `deployment/college_pc_runkit/*` (setup, validation, run scripts, requirements)
- Empty `data/` structure placeholders

## What you must add after unzipping
- `data/mimic_database.duckdb`
- `data/raw/MIMIC-IV-ECG-1.0/files/...` waveform tree
- optional: `data/processed/cohort_master.parquet`

See `data/README_DATA_PLACEMENT.md` for placement details.

## Recommended usage on college PC
1. Open PowerShell in the extracted folder root (the folder that contains `scripts/`, `data/`, and `deployment/`).
2. Setup environment:
   - `powershell -ExecutionPolicy Bypass -File deployment/college_pc_runkit/setup_college_env.ps1 -Gpu`
   - If the remote PC has no CUDA GPU, use CPU setup instead:
     - `powershell -ExecutionPolicy Bypass -File deployment/college_pc_runkit/setup_college_env.ps1`
3. Setup MATLAB toolboxes (if not already configured):
   - `powershell -ExecutionPolicy Bypass -File deployment/college_pc_runkit/setup_matlab_toolboxes.ps1 -CloneRepos`
4. Validate setup and data placement:
   - `.\cuda_env\Scripts\python.exe deployment/college_pc_runkit/verify_setup.py`
   - `.\cuda_env\Scripts\python.exe deployment/college_pc_runkit/validate_data_placement.py`
5. Run 100-sample extraction:
   - `powershell -ExecutionPolicy Bypass -File deployment/college_pc_runkit/run_hybrid_100.ps1`
   - CPU-safe run (if CUDA is unavailable):
     - `powershell -ExecutionPolicy Bypass -File deployment/college_pc_runkit/run_hybrid_100_cpu.ps1`

## Portability checklist for a new remote PC
- Run all commands from this bundle root (do not run from `deployment/college_pc_runkit/`).
- Confirm `matlab` command is available in PATH before running extraction.
- Keep the data tree names exactly as documented in `data/README_DATA_PLACEMENT.md`.
- Ensure write permission to `data/processed/`.
