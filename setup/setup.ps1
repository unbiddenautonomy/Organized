param(
    [string]$Python = 'py',
    [string]$FoInterface = 'auto',
    [string]$CmInterface = 'auto',
    [string]$ContractUrl = '',
    [switch]$SkipPackages
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$venvPython = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    if ($Python -eq 'py') {
        & py -3.13 -m venv (Join-Path $repo '.venv')
    } else {
        & $Python -m venv (Join-Path $repo '.venv')
    }
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.13 virtual environment creation failed' }
}

if (-not $SkipPackages) {
    & $venvPython -m pip install -r (Join-Path $repo 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
}

$configure = @((Join-Path $repo 'setup\voltr.py'), 'configure', '--fo-interface', $FoInterface,
               '--cm-interface', $CmInterface)
if ($ContractUrl) { $configure += @('--contract-url', $ContractUrl) }
& $venvPython @configure
if ($LASTEXITCODE -ne 0) { throw 'Machine configuration failed' }

& $venvPython (Join-Path $repo 'setup\voltr.py') preflight --pipeline ivrv
if ($LASTEXITCODE -ne 0) { throw 'IVRV offline preflight failed' }
Write-Host 'Offline setup passed. Configure ERP_AUTH_TOKEN and verify contract/feed access before installing the task.'
