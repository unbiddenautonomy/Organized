param([switch]$SkipPackages)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$inferencePython = Join-Path $repo '.venv\Scripts\python.exe'
$volpredPython = Join-Path $repo '.venv-volpred\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $inferencePython)) {
    throw 'Run setup/setup.ps1 first to create the inference environment'
}
if (-not (Test-Path -LiteralPath $volpredPython)) {
    & py -3.13 -m venv (Join-Path $repo '.venv-volpred')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the VolPred environment' }
}
if (-not $SkipPackages) {
    & $inferencePython -m pip install -r (Join-Path $repo 'research\v2\requirements-training.txt')
    if ($LASTEXITCODE -ne 0) { throw 'V2 research dependency installation failed' }
    & $volpredPython -m pip install -r (Join-Path $repo 'research\volpred\requirements-vol-research.txt')
    if ($LASTEXITCODE -ne 0) { throw 'VolPred dependency installation failed' }
}
& $inferencePython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Inference/V2 dependency check failed' }
& $volpredPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'VolPred dependency check failed' }
Write-Host 'Research environments are ready. Large historical inputs remain external.'
