param(
    [string]$TaskName = 'VolTr-PaperTrading-Clean',
    [string]$StartTime = '08:55',
    [switch]$Apply
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$app = $repo  # the service runs from the repository root
$python = Join-Path $repo '.venv\Scripts\python.exe'
$config = Join-Path $repo 'config\live_config.json'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run setup\setup.ps1 first' }
if (-not $Apply) {
    Write-Host "Planned task: $TaskName at $StartTime daily; Python=$python; working directory=$app"
    Write-Host 'Pass -Apply to register it after the offline preflight and ERP token check.'
    exit 0
}
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw "Task $TaskName already exists; choose a new -TaskName or inspect the existing task"
}
if (-not $env:ERP_AUTH_TOKEN) { throw 'ERP_AUTH_TOKEN is missing from this user session' }
& $python (Join-Path $repo 'setup\voltr.py') preflight --pipeline ivrv
if ($LASTEXITCODE -ne 0) { throw 'IVRV preflight failed; task was not registered' }

$action = New-ScheduledTaskAction -Execute $python -Argument ('-m pipelines.ivrv.live --config "{0}"' -f $config) -WorkingDirectory $app
$trigger = New-ScheduledTaskTrigger -Daily -At $StartTime
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 10)
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description 'Portable BANKNIFTY IVRV paper trading' | Out-Null
Write-Host "Registered $TaskName for this user. The task was not started."
