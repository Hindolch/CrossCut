param(
    [string]$Python = "$PSScriptRoot\..\.venv-local\Scripts\python.exe",
    [string]$RunId = 'exp002-astra-seed0',
    [string]$Entity = 'nileshsarkar-ai',
    [string]$Project = 'crosscut-llm4teach'
)
$ErrorActionPreference = 'Stop'
if ($RunId -notmatch '^[a-zA-Z0-9_-]{1,80}$' -or $Entity -notmatch '^[a-zA-Z0-9_-]+$' -or $Project -notmatch '^[a-zA-Z0-9_-]+$') {
    throw 'Run ID, entity, and project must be simple identifiers.'
}
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$output = Join-Path $workspace 'data\local\exp002'
New-Item -ItemType Directory -Force -Path $output | Out-Null
if (Test-Path -LiteralPath (Join-Path $output 'STOP')) { throw 'Remove the STOP file before intentionally restarting the monitor.' }
$secret = Read-Host 'W&B API key (used only by the local child process)' -AsSecureString
$pointer = [IntPtr]::Zero
try {
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
    $env:WANDB_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    $child = Start-Process -FilePath $Python -ArgumentList @('-m', 'crosscut.student_monitor', '--run-id', $RunId, '--entity', $Entity, '--project', $Project) -WorkingDirectory $workspace -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $output 'monitor.stdout.log') -RedirectStandardError (Join-Path $output 'monitor.stderr.log')
    Write-Output "Local student monitor PID: $($child.Id)"
} finally {
    Remove-Item Env:\WANDB_API_KEY -ErrorAction SilentlyContinue
    if ($pointer -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
    $secret.Dispose()
}
