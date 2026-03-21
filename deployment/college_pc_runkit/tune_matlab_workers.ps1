param(
  [string]$MatlabExe = "matlab",
  [int]$Workers = 0,
  [int]$PowerfulCpuThreshold = 12,
  [int]$ReserveLogicalCores = 2,
  [int]$MaxWorkersCap = 24,
  [string]$ClusterProfile = "Processes",
  [string]$LogPath = "deployment/college_pc_runkit/logs/matlab_worker_tuning.log"
)

$ErrorActionPreference = 'Stop'

if (!(Test-Path .\scripts\matlab\extract_mimic_ecgdeli_features.m)) {
  throw "Run this script from project root (folder containing scripts/, data/, deployment/)."
}

$matlabCommand = Get-Command $MatlabExe -ErrorAction SilentlyContinue
if (-not $matlabCommand) {
  throw "MATLAB executable '$MatlabExe' was not found in PATH. Pass -MatlabExe with a valid executable path."
}

$logicalCores = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
if (-not $logicalCores -or $logicalCores -lt 1) {
  $logicalCores = [int]$env:NUMBER_OF_PROCESSORS
}
if (-not $logicalCores -or $logicalCores -lt 1) {
  throw "Could not determine logical CPU core count."
}

$autoCandidate = [Math]::Max(1, $logicalCores - [Math]::Max(0, $ReserveLogicalCores))
if ($MaxWorkersCap -gt 0) {
  $autoCandidate = [Math]::Min($autoCandidate, $MaxWorkersCap)
}

$requestedWorkers = 0
$applyIncrease = $false
if ($Workers -gt 0) {
  $requestedWorkers = $Workers
  if ($MaxWorkersCap -gt 0 -and $requestedWorkers -gt $MaxWorkersCap) {
    Write-Warning "Manual -Workers value ($requestedWorkers) exceeds MaxWorkersCap ($MaxWorkersCap); clamping to $MaxWorkersCap."
    $requestedWorkers = $MaxWorkersCap
  }
  $applyIncrease = $true
  Write-Host "Manual worker request detected: $requestedWorkers"
} elseif ($logicalCores -ge $PowerfulCpuThreshold) {
  $requestedWorkers = $autoCandidate
  $applyIncrease = $true
  Write-Host "Powerful CPU detected ($logicalCores logical cores). Requesting worker increase target: $requestedWorkers"
} else {
  Write-Host "CPU not considered powerful enough for auto-increase ($logicalCores < $PowerfulCpuThreshold). Keeping current MATLAB profile worker count."
}

$matlabBatchTemplate = @'
requested = __WORKERS__;
applyIncrease = __APPLY_INCREASE__;
profileName = '__PROFILE__';
try
  c = parcluster(profileName);
catch ME
  fprintf(2, 'FATAL: Could not load cluster profile %s: %s\n', profileName, ME.message);
  exit(2);
end
fprintf('Using cluster profile: %s\n', profileName);
fprintf('Before update NumWorkers=%d\n', c.NumWorkers);
desired = c.NumWorkers;
try
  if applyIncrease && requested >= 1
    desired = max(c.NumWorkers, requested);
  end
  if desired > c.NumWorkers
    c.NumWorkers = desired;
    saveProfile(c);
    fprintf('Updated profile NumWorkers to %d\n', c.NumWorkers);
  else
    fprintf('No profile increase applied (requested=%d, current=%d)\n', requested, c.NumWorkers);
  end
catch ME
  fprintf(2, 'FATAL: Could not update profile: %s\n', ME.message);
  exit(3);
end
c = parcluster(profileName);
fprintf('After reload NumWorkers=%d\n', c.NumWorkers);
if c.NumWorkers ~= desired
  fprintf(2, 'FATAL: Profile save verification failed (expected=%d, got=%d)\n', desired, c.NumWorkers);
  exit(4);
end
poolTarget = c.NumWorkers;
try
  p = gcp('nocreate');
  if ~isempty(p), delete(p); end
  % Smoke-test only: verify that requested worker count can actually start.
  parpool(profileName, poolTarget);
  p = gcp('nocreate');
  fprintf('parpool %s %d SUCCESS, actual workers=%d\n', profileName, poolTarget, p.NumWorkers);
  delete(p);
catch ME
  fprintf(2, 'FATAL: parpool %s %d FAILED: %s\n', profileName, poolTarget, ME.message);
  exit(5);
end
exit(0);
'@

$profileEscaped = $ClusterProfile.Replace("'", "''")
$applyIncreaseInt = if ($applyIncrease) { "1" } else { "0" }
$batchCommand = $matlabBatchTemplate.Replace("__WORKERS__", [string]$requestedWorkers)
$batchCommand = $batchCommand.Replace("__APPLY_INCREASE__", $applyIncreaseInt)
$batchCommand = $batchCommand.Replace("__PROFILE__", $profileEscaped)

$logDir = Split-Path -Parent $LogPath
if ($logDir -and !(Test-Path $logDir)) {
  New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

& $MatlabExe -batch $batchCommand 2>&1 | Tee-Object -FilePath $LogPath

if ($LASTEXITCODE -ne 0) {
  throw "MATLAB worker tuning failed with exit code $LASTEXITCODE. See log: $LogPath"
}

Write-Host "MATLAB worker tuning completed. Log: $LogPath"