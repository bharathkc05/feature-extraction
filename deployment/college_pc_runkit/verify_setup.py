from pathlib import Path
import importlib

BASE = Path.cwd()
required_paths = [
    BASE / "scripts" / "phase_d_hybrid_192_feature_extraction.py",
    BASE / "scripts" / "matlab" / "extract_mimic_ecgdeli_features.m",
    BASE / "data" / "mimic_database.duckdb",
    BASE / "data" / "raw" / "MIMIC-IV-ECG-1.0",
]

print("=== PATH CHECK ===")
missing = [p for p in required_paths if not p.exists()]
if missing:
    for p in missing:
        print(f"MISSING: {p}")
else:
    print("All required core paths found.")

print("\n=== PYTHON PACKAGE CHECK ===")
packages = ["numpy", "pandas", "duckdb", "neurokit2", "wfdb", "torch", "tqdm", "scipy"]
for pkg in packages:
    try:
        mod = importlib.import_module(pkg)
        print(f"OK: {pkg}=={getattr(mod, '__version__', 'n/a')}")
    except Exception as exc:
        print(f"FAIL: {pkg} ({exc})")

print("\n=== CUDA CHECK ===")
try:
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU0: {torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"Torch/CUDA check failed: {exc}")

print("\n=== DONE ===")
