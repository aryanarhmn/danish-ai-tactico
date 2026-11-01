# Danish AI-Tactico - one-time setup on Windows.
#
# Run from this folder in PowerShell:
#     powershell -ExecutionPolicy Bypass -File setup.ps1
#
# Installs Python 3.11 and Git if missing, creates .venv, installs pinned packages,
# points MTAH_DATA_DIR at C:\mtah-data and scores the four baselines on the public 5 minutes.

$ErrorActionPreference = "Stop"
$DataDir = "C:\mtah-data"

function Have($cmd) { return [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

function HavePy311 {
    if (-not (Have "py")) { return $false }
    & py -3.11 -c "import sys" 2>$null
    return ($LASTEXITCODE -eq 0)
}

if (-not (HavePy311)) {
    Write-Host "Installing Python 3.11 ..."
    winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
}
if (-not (Have "git")) {
    Write-Host "Installing Git ..."
    winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv ..."
    if (-not (HavePy311)) {
        Write-Host "Python 3.11 is still not found. Close PowerShell, open a new one and run setup.ps1 again."
        exit 1
    }
    py -3.11 -m venv .venv
}
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Could not create .venv. Delete the .venv folder and run setup.ps1 again."
    exit 1
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

if (-not (Test-Path $DataDir)) { New-Item -ItemType Directory -Path $DataDir | Out-Null }
[Environment]::SetEnvironmentVariable("MTAH_DATA_DIR", $DataDir, "User")
$env:MTAH_DATA_DIR = $DataDir

Write-Host ""
Write-Host "Scoring the baselines on the public 5 minutes (needs no match data) ..."
& .\.venv\Scripts\python.exe evaluation\run_evaluation.py --no-detail

Write-Host ""
Write-Host "Done. Unpack the match data so that this file exists:"
Write-Host "    $DataDir\tactical\BAR-ATH_tactical_atd.txt"
Write-Host "Then, in a new PowerShell window in this folder:"
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host "    python run.py"
