param(
    [Parameter(Mandatory = $true)][string]$SshHost,
    [int]$SshPort = 22,
    [int]$LocalPort = 8765,
    [int]$RemotePort = 8765
)
$ErrorActionPreference = 'Stop'
# Use a configured SSH host alias if an identity file or proxy is needed.
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -p $SshPort -L "127.0.0.1:${LocalPort}:127.0.0.1:${RemotePort}" $SshHost
if ($LASTEXITCODE -ne 0) { throw 'SSH tunnel stopped.' }
