param([switch]$Detached)
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoPath
$pythonPath = Join-Path $repoPath '.venv-local\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run scripts/setup-local.ps1 first.' }
& $pythonPath -m crosscut.client health
if ($LASTEXITCODE -ne 0) { throw 'The Crafter server must be reachable before autoplay starts.' }
New-Item -ItemType Directory -Path 'data' -Force | Out-Null
$monitorProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m', 'crosscut.monitor') -WorkingDirectory $repoPath -WindowStyle Hidden -PassThru -RedirectStandardOutput 'data\monitor.stdout.log' -RedirectStandardError 'data\monitor.stderr.log'
Start-Sleep -Seconds 2
if ($monitorProcess.HasExited) { throw 'Monitor failed; inspect data/monitor.stderr.log.' }
if ($Detached) {
    $playerProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m', 'crosscut.autoplay') -WorkingDirectory $repoPath -WindowStyle Hidden -PassThru -RedirectStandardOutput 'data\autoplay.stdout.log' -RedirectStandardError 'data\autoplay.stderr.log'
    @{ monitor = $monitorProcess.Id; autoplay = $playerProcess.Id } | ConvertTo-Json | Set-Content -LiteralPath 'data\local-processes.json'
    Write-Output "Local autoplay PID $($playerProcess.Id); monitor PID $($monitorProcess.Id)."
} else {
    try { & $pythonPath -m crosscut.autoplay }
    finally { Write-Output "Monitor continues as PID $($monitorProcess.Id); logs are in data/." }
}
