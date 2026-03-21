param(
  [switch]$Gpu
)

$ErrorActionPreference = 'Stop'

if (!(Test-Path .\deployment\college_pc_runkit\requirements-cpu.txt)) {
  throw "Run this script from the project root (folder containing deployment/, scripts/, data/)."
}

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
  throw "Python was not found in PATH. Install Python 3.10+ and retry."
}

if (!(Test-Path .\cuda_env\Scripts\python.exe)) {
  & python -m venv cuda_env
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to create virtual environment at .\\cuda_env"
  }
}

.\cuda_env\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
  throw "Failed to upgrade pip/setuptools/wheel in .\\cuda_env"
}

if ($Gpu) {
  Write-Host "Installing GPU dependencies (CUDA 12.1 wheels)..."
  .\cuda_env\Scripts\python.exe -m pip install -r deployment/college_pc_runkit/requirements-gpu-cu121.txt --extra-index-url https://download.pytorch.org/whl/cu121
} else {
  Write-Host "Installing CPU dependencies..."
  .\cuda_env\Scripts\python.exe -m pip install -r deployment/college_pc_runkit/requirements-cpu.txt
}
if ($LASTEXITCODE -ne 0) {
  throw "Dependency installation failed. Check network/proxy and requirement compatibility."
}

.\cuda_env\Scripts\python.exe deployment/college_pc_runkit/verify_setup.py
if ($LASTEXITCODE -ne 0) {
  throw "verify_setup.py failed. Resolve reported issues and retry."
}

.\cuda_env\Scripts\python.exe deployment/college_pc_runkit/validate_data_placement.py
if ($LASTEXITCODE -ne 0) {
  throw "validate_data_placement.py failed. Place required data files and retry."
}

Write-Host "Environment setup complete."
