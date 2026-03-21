# Data Placement (Empty in Bundle)

This bundle is intentionally shipped WITHOUT real data.

Place these later after unzipping:

Required:
- `data/mimic_database.duckdb`
- `data/raw/MIMIC-IV-ECG-1.0/files/...` (full waveform tree)

Optional:
- `data/processed/cohort_master.parquet`

Expected waveform structure pattern:
- `data/raw/MIMIC-IV-ECG-1.0/files/pXXXX/pXXXXXXXX/sXXXXXXXX/`

Do not remove folder structure placeholders.

Quick validation from bundle root:
- `.\cuda_env\Scripts\python.exe deployment/college_pc_runkit/validate_data_placement.py`
