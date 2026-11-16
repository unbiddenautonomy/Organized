$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$v2 = Join-Path $repo 'research\v2'
$volpred = Join-Path $repo 'research\volpred'
$inferencePython = Join-Path $repo '.venv\Scripts\python.exe'
$volpredPython = Join-Path $repo '.venv-volpred\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $inferencePython) -or
    -not (Test-Path -LiteralPath $volpredPython)) {
    throw 'Run setup/setup.ps1 and setup/setup_research.ps1 first'
}

$previousPythonPath = $env:PYTHONPATH
try {
    $cases = @(
        @('research\combined_atm_iv_rv_convergence_v2', 'research\combined_atm_iv_rv_convergence_v2\tests\test_feature_pipeline.py'),
        @('research\fixed_strike_iv_rv_convergence', 'research\fixed_strike_iv_rv_convergence\tests\test_feature_pipeline.py'),
        @('research\fixed_strike_iv_120m', 'research\fixed_strike_iv_120m\tests\test_fixed_strike_pipeline.py'),
        @('models\rv_decline_faster_than_iv_120m', 'models\rv_decline_faster_than_iv_120m\test_train.py')
    )
    foreach ($case in $cases) {
        $env:PYTHONPATH = Join-Path $v2 $case[0]
        & $inferencePython -m pytest (Join-Path $v2 $case[1]) -q --disable-warnings
        if ($LASTEXITCODE -ne 0) { throw "V2 research test failed: $($case[1])" }
    }
    $env:PYTHONPATH = "$volpred;$volpred\research"
    Push-Location -LiteralPath $volpred
    try {
        & $volpredPython -m pytest tests -q --disable-warnings
        if ($LASTEXITCODE -ne 0) { throw 'VolPred research tests failed' }
    } finally {
        Pop-Location
    }
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
Write-Host 'V2 and VolPred research tests passed.'
