$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoPath
$sessionSecret = Read-Host 'W&B API key (session only)' -AsSecureString
$secretPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sessionSecret)
try {
    $env:WANDB_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($secretPointer)
    $env:PYTHONUTF8 = '1'
    $monitorProcess = Start-Process -FilePath (Join-Path $repoPath '.venv-local\Scripts\python.exe') -ArgumentList @('-m', 'crosscut.monitor') -WorkingDirectory $repoPath -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $repoPath 'data\monitor.stdout.log') -RedirectStandardError (Join-Path $repoPath 'data\monitor.stderr.log')
    $monitorProcess.Id | Set-Content -LiteralPath (Join-Path $repoPath 'data\monitor.pid')
    Write-Output "Detached W&B and Git monitor started: PID $($monitorProcess.Id)"
} finally {
    Remove-Item Env:WANDB_API_KEY -ErrorAction SilentlyContinue
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($secretPointer)
    $sessionSecret.Dispose()
}
