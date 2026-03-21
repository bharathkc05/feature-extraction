from pathlib import Path
import sys

ROOT = Path.cwd()

required_paths = {
    "DuckDB": ROOT / "data" / "mimic_database.duckdb",
    "Waveform root": ROOT / "data" / "raw" / "MIMIC-IV-ECG-1.0",
    "Extractor script": ROOT / "scripts" / "phase_d_hybrid_192_feature_extraction.py",
    "MATLAB extractor": ROOT / "scripts" / "matlab" / "extract_mimic_ecgdeli_features.m",
}

print("=== DATA PLACEMENT VALIDATION ===")

missing = []
for label, path in required_paths.items():
    if path.exists():
        print(f"OK: {label}: {path}")
    else:
        print(f"MISSING: {label}: {path}")
        missing.append(label)

wave_root = required_paths["Waveform root"]

structure_ok = False
sample_subject_dirs = []
sample_patient_dirs = []
sample_study_dirs = []

if wave_root.exists():
    files_root = wave_root / "files"
    if files_root.exists():
        sample_subject_dirs = [p for p in files_root.glob("p*") if p.is_dir()][:5]
        for subj in sample_subject_dirs:
            sample_patient_dirs.extend([p for p in subj.glob("p*") if p.is_dir()][:2])
        for patient in sample_patient_dirs[:8]:
            sample_study_dirs.extend([s for s in patient.glob("s*") if s.is_dir()][:2])

        structure_ok = len(sample_subject_dirs) > 0 and len(sample_patient_dirs) > 0 and len(sample_study_dirs) > 0

        print("\n=== WAVEFORM TREE CHECK ===")
        print(f"files root: {files_root}")
        print(f"subject-level dirs found (files/p*): {len(sample_subject_dirs)}")
        print(f"patient-level dirs found (files/p*/p*): {len(sample_patient_dirs)}")
        print(f"study-level dirs found (files/p*/p*/s*): {len(sample_study_dirs)}")
    else:
        print(f"MISSING: expected waveform subfolder: {files_root}")

if not structure_ok:
    print("\nFAIL: MIMIC waveform structure does not look correct.")
    print("Expected pattern under waveform root: files/pXXXX/pXXXXXXXX/sXXXXXXXX/")
    sys.exit(2)

if missing:
    print("\nFAIL: Missing required core paths.")
    sys.exit(1)

print("\nPASS: Data placement looks correct for running the hybrid pipeline.")
sys.exit(0)
