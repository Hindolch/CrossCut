param([int]$Port = 8767)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$teacherPython = Join-Path $projectRoot '.venv-local\Scripts\python.exe'
$teacherData = Join-Path $projectRoot 'data\local\exp002\teacher'
New-Item -ItemType Directory -Force -Path $teacherData | Out-Null
if (Test-Path -LiteralPath (Join-Path $teacherData 'STOP')) {
    throw 'Teacher STOP file exists. Remove it explicitly before starting.'
}
$teacherProcess = Start-Process -FilePath $teacherPython -ArgumentList @('-m', 'crosscut.astra_worker', '--port', $Port) -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $teacherData 'worker.stdout.log') -RedirectStandardError (Join-Path $teacherData 'worker.stderr.log')
$teacherProcess.Id | Set-Content -LiteralPath (Join-Path $teacherData 'worker.pid')
Write-Output "Local Astra teacher started: PID $($teacherProcess.Id), http://127.0.0.1:$Port"
