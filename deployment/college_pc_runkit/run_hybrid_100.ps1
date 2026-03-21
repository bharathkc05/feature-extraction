param(
  [switch]$SkipMatlabWorkerTuning
)

$ErrorActionPreference = 'Stop'

if (!(Test-Path .\scripts\phase_d_hybrid_192_feature_extraction.py)) {
  throw "Run this script from project root (folder containing scripts/, data/, deployment/)."
}

if (!(Test-Path .\cuda_env\Scripts\python.exe)) {
  throw "Python environment not found at .\\cuda_env. Run deployment/college_pc_runkit/setup_college_env.ps1 first."
}

if (-not $SkipMatlabWorkerTuning) {
  $tunerPath = ".\deployment\college_pc_runkit\tune_matlab_workers.ps1"
  if (Test-Path $tunerPath) {
    Write-Host "Auto-tuning MATLAB Processes workers before extraction..."
    try {
      $tunerOutput = powershell -ExecutionPolicy Bypass -File $tunerPath 2>&1
      if ($tunerOutput) {
        $tunerOutput | ForEach-Object { Write-Host $_ }
      }

      $workersFromTuner = $null
      foreach ($line in $tunerOutput) {
        if ($line -match 'After reload NumWorkers=(\d+)') {
          $workersFromTuner = [int]$matches[1]
        }
      }

      if ($workersFromTuner -and $workersFromTuner -gt 0) {
        $env:MATLAB_ECGDELI_WORKERS = [string]$workersFromTuner
        Write-Host "Exported MATLAB_ECGDELI_WORKERS=$($env:MATLAB_ECGDELI_WORKERS) for this run."
      } else {
        Write-Warning "Could not parse tuned MATLAB NumWorkers from tuner output; Python will use its internal fallback worker logic."
      }
    } catch {
      Write-Warning "MATLAB worker auto-tuning failed: $($_.Exception.Message). Continuing with extraction."
    }
  } else {
    Write-Warning "MATLAB worker tuner script not found at $tunerPath. Continuing without auto-tuning."
  }
}

# Optional cleanup of stale processes
Get-CimInstance Win32_Process |
  Where-Object { $_.Name -match '^python\.exe$' -and $_.CommandLine -match 'phase_d_hybrid_192_feature_extraction.py' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Get-Process |
  Where-Object { $_.ProcessName -match '^MATLAB$|^matlab$|^matlabwindowhelper$' } |
  Stop-Process -Force -ErrorAction SilentlyContinue

.\cuda_env\Scripts\python.exe scripts/phase_d_hybrid_192_feature_extraction.py `
  --limit 100 `
  --run-matlab-ecgdeli `
  --require-ecgdeli `
  --parallel-processes 1 `
  --gpu-stats `
  --checkpoint-every 25 `
  --matlab-timeout-seconds 3600 `
  --output-csv data/processed/test100_hybrid_192.csv `
  --output-parquet data/processed/test100_hybrid_192.parquet

if ($LASTEXITCODE -ne 0) {
  throw "Hybrid extraction failed. If this machine has no CUDA GPU, run with --gpu-allow-cpu-fallback using the README fallback command."
}

Write-Host "Run completed successfully."
