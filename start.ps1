param([switch]$StartNotebook, [switch]$Share, [double]$Hours = 3)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create Python virtual environment' }
}
& ./.venv/Scripts/python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
$launchArgs = @('app.py', '--hours', $Hours)
if ($StartNotebook) { $launchArgs += '--start' }
if ($Share) { $launchArgs += '--share' }
& ./.venv/Scripts/python.exe @launchArgs
