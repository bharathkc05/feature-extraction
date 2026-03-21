param(
  [string]$MatlabExe = "matlab",
  [switch]$CloneRepos,
  [string]$EcgdeliRepoUrl = "https://github.com/KIT-IBT/ECGdeli.git",
  [string]$EcgdeliTag = "v1.1",
  [string]$WfdbZipUrl = "https://physionet.org/physiotools/matlab/wfdb-app-matlab/wfdb-app-toolbox-0-10-0.zip",
  [string]$EcgdeliPath = "third_party/ECGDeli",
  [string]$WfdbPath = "third_party/wfdb"
)

$ErrorActionPreference = 'Stop'

if (!(Test-Path .\scripts\matlab\extract_mimic_ecgdeli_features.m)) {
  throw "Run this script from project root (folder containing scripts/, data/, deployment/)."
}

$matlabCommand = Get-Command $MatlabExe -ErrorAction SilentlyContinue
if (-not $matlabCommand) {
  throw "MATLAB executable '$MatlabExe' was not found in PATH. Pass -MatlabExe with a valid executable path."
}

function To-MatlabPath([string]$p) {
  return ((Resolve-Path $p).Path -replace "\\", "/")
}

if (!(Test-Path $WfdbPath)) {
  New-Item -ItemType Directory -Path $WfdbPath -Force | Out-Null
}

if ($CloneRepos) {
  $gitCommand = Get-Command git -ErrorAction SilentlyContinue
  if (-not $gitCommand) {
    throw "git was not found in PATH, but -CloneRepos was requested. Install git or omit -CloneRepos."
  }

  if (!(Test-Path $EcgdeliPath)) {
    git clone $EcgdeliRepoUrl $EcgdeliPath
  }

  if (Test-Path (Join-Path $EcgdeliPath ".git")) {
    if (![string]::IsNullOrWhiteSpace($EcgdeliTag)) {
      git -C $EcgdeliPath fetch --tags
      git -C $EcgdeliPath checkout $EcgdeliTag
      Write-Host "ECGDeli checked out at tag: $EcgdeliTag"
    }
  } else {
    Write-Host "ECGDeli path exists but is not a git repo: $EcgdeliPath"
    Write-Host "Skipping tag checkout; ensure this folder matches release $EcgdeliTag."
  }
}

# Official WFDB MATLAB toolbox installation (PhysioNet ZIP method)
$wfdbInstallRoot = To-MatlabPath $WfdbPath
$wfdbZipEscaped = $WfdbZipUrl.Replace("'", "''")
$wfdbInstallCmd = @"
cd('$wfdbInstallRoot');
old_path = which('rdsamp');
if (~isempty(old_path))
  rmpath(old_path(1:end-8));
end
wfdb_url = '$wfdbZipEscaped';
[~,~] = urlwrite(wfdb_url,'wfdb-app-toolbox-0-10-0.zip');
unzip('wfdb-app-toolbox-0-10-0.zip');
cd mcode;
addpath(pwd);
savepath;
fprintf('WFDB toolbox installed via official PhysioNet ZIP flow.\\n');
fprintf('WFDB rdsamp path: %s\\n', which('rdsamp'));
"@

& $MatlabExe -batch $wfdbInstallCmd

if (!(Test-Path $EcgdeliPath)) {
  throw "ECGDeli path not found: $EcgdeliPath`nEither clone with -CloneRepos -EcgdeliRepoUrl <url> or place ECGDeli manually."
}

$wfdbMatlabPath = Join-Path $WfdbPath "mcode"
if (!(Test-Path $wfdbMatlabPath)) {
  throw "WFDB MATLAB mcode path not found: $wfdbMatlabPath"
}

$ecgdeliMat = To-MatlabPath $EcgdeliPath
$wfdbMat = To-MatlabPath $wfdbMatlabPath

$cmd = @"
addpath(genpath('$ecgdeliMat'));
addpath(genpath('$wfdbMat'));
savepath;
required = {'rdsamp','Annotate_ECG_Multi','ExtractAmplitudeFeaturesFromFPT','ExtractIntervalFeaturesFromFPT','ECG_High_Low_Filter','Notch_Filter','Isoline_Correction','ECG_Baseline_Removal'};
for i=1:numel(required)
  if exist(required{i}, 'file') ~= 2
    error('Missing required MATLAB function: %s', required{i});
  end
end
fprintf('MATLAB toolbox preflight passed.\n');
fprintf('WFDB rdsamp: %s\n', which('rdsamp'));
fprintf('ECGDeli Annotate_ECG_Multi: %s\n', which('Annotate_ECG_Multi'));
"@

& $MatlabExe -batch $cmd
Write-Host "MATLAB toolbox setup complete."
