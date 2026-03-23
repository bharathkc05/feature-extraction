from __future__ import annotations

import argparse
import sys
from pathlib import Path


SUSPICIOUS_EXTENSIONS = {
    ".atr",
    ".dat",
    ".edf",
    ".hea",
    ".hea-",
    ".mat",
    ".rec",
    ".trigger",
}

SUSPICIOUS_MARKERS = (
    b"<html",
    b"<!doctype html",
    b"301 moved permanently",
    b"nginx",
)


def resolve_default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def looks_like_html_artifact(file_path: Path) -> bool:
    try:
        with file_path.open("rb") as handle:
            chunk = handle.read(4096).lower()
    except OSError:
        return False

    return any(marker in chunk for marker in SUSPICIOUS_MARKERS)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate WFDB files are not HTML/redirect artifacts."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=resolve_default_root() / "third_party" / "wfdb" / "database",
        help="Path to WFDB dataset root (default: third_party/wfdb/database)",
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=25,
        help="Maximum suspicious files to print (default: 25)",
    )
    args = parser.parse_args()

    dataset_root = args.path.resolve()
    if not dataset_root.exists():
        print(f"[ERROR] Path does not exist: {dataset_root}")
        return 2

    if not dataset_root.is_dir():
        print(f"[ERROR] Path is not a directory: {dataset_root}")
        return 2

    total_checked = 0
    suspicious_files: list[Path] = []

    for file_path in dataset_root.rglob("*"):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        if suffix not in SUSPICIOUS_EXTENSIONS and file_path.name.lower().endswith(".hea-") is False:
            continue

        total_checked += 1
        if looks_like_html_artifact(file_path):
            suspicious_files.append(file_path)

    print(f"[INFO] Dataset root: {dataset_root}")
    print(f"[INFO] Checked files: {total_checked}")
    print(f"[INFO] Suspicious files: {len(suspicious_files)}")

    if suspicious_files:
        print("[ERROR] Detected HTML/redirect artifacts in WFDB files.")
        for file_path in suspicious_files[: max(args.max_report, 0)]:
            rel = file_path.relative_to(dataset_root)
            print(f"  - {rel}")

        if len(suspicious_files) > args.max_report:
            remaining = len(suspicious_files) - args.max_report
            print(f"  ... and {remaining} more")

        return 1

    print("[OK] No HTML/redirect artifacts detected in scanned WFDB files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
