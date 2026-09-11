param([string]$Python = "$PSScriptRoot\..\.venv-local\Scripts\python.exe")
$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$output = Join-Path $workspace 'data\local\exp002'
$key = Join-Path $workspace 'data\session-access-exp002\id_ed25519'
if (-not (Test-Path -LiteralPath $key -PathType Leaf)) { throw "Missing SSH key: $key" }
New-Item -ItemType Directory -Force -Path $output | Out-Null
if (Test-Path -LiteralPath (Join-Path $output 'TUNNEL_STOP')) {
    throw 'Remove TUNNEL_STOP before intentionally restarting the tunnel.'
}
$child = Start-Process -FilePath $Python -ArgumentList @('-m', 'crosscut.experiment_tunnel') -WorkingDirectory $workspace -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $output 'tunnel.stdout.log') -RedirectStandardError (Join-Path $output 'tunnel.stderr.log')
Write-Output "Experiment tunnel supervisor PID: $($child.Id)"
