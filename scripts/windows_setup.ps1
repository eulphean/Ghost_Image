# One-time setup on 64-bit Windows.
# Creates .venv, installs the pinned Python packages, and downloads the
# segmentation models. Safe to re-run.
#
# Usage, from the project folder:
#   powershell -ExecutionPolicy Bypass -File scripts\windows_setup.ps1
# or double-click scripts\windows_setup.bat

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not [Environment]::Is64BitOperatingSystem) {
    Write-Error "64-bit Windows is required. MediaPipe has no 32-bit Windows wheel."
}

function Test-Python {
    param([string]$Exe, [string[]]$Prefix)
    & $Exe @Prefix -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) and sys.maxsize > 2**32 else 1)"
    return $LASTEXITCODE -eq 0
}

$python = $null
$prefix = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    if (Test-Python "py" @("-3")) {
        $python = "py"
        $prefix = @("-3")
    }
}
if (-not $python -and (Get-Command python -ErrorAction SilentlyContinue)) {
    if (Test-Python "python" @()) {
        $python = "python"
    }
}
if (-not $python) {
    Write-Error "Python 3.11 or newer (64-bit) was not found. Install it from https://www.python.org/downloads/ and enable 'Add python.exe to PATH'."
}

$version = & $python @prefix -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "==> System Python is $version"

if (-not (Test-Path ".venv")) {
    Write-Host "==> Creating virtual environment in .venv"
    & $python @prefix -m venv .venv
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$venvPython = Join-Path ".venv" "Scripts\python.exe"
Write-Host "==> Installing Python dependencies"
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Downloading segmentation models"
New-Item -ItemType Directory -Force -Path "models" | Out-Null
$models = @{
    "models\selfie_segmenter.tflite" = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
    "models\selfie_multiclass_256x256.tflite" = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
}
foreach ($dest in $models.Keys) {
    Write-Host "==> $dest"
    Invoke-WebRequest -Uri $models[$dest] -OutFile $dest
}

Write-Host ""
Write-Host "Done. From this folder, run:"
Write-Host "  .\.venv\Scripts\python.exe -m ghost_image"
Write-Host "Or double-click scripts\windows_run.bat"
