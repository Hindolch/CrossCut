$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoPath
if (-not (Test-Path -LiteralPath '.venv-local\Scripts\python.exe')) {
    py -3.13 -m venv .venv-local
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.13 is required for this setup script.' }
}
& '.\.venv-local\Scripts\python.exe' -m pip install -r requirements-local.txt
if ($LASTEXITCODE -ne 0) { throw 'Local dependency installation failed.' }
if (-not (Test-Path -LiteralPath 'config.local.json')) {
    Copy-Item -LiteralPath 'config.example.json' -Destination 'config.local.json'
}
& '.\.venv-local\Scripts\python.exe' -m crosscut.autoplay --dry-run
