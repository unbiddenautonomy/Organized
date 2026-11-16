[CmdletBinding()]
param(
    [switch]$SkipFullDataAudit
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

if (-not $SkipFullDataAudit) {
    py -3.13 tests/data_audit_scan.py
    if ($LASTEXITCODE -ne 0) { throw "Data audit failed with exit code $LASTEXITCODE" }
}

py -3.13 -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Test suite failed with exit code $LASTEXITCODE" }

Write-Host "Research integrity gate reproduced successfully."
Write-Host "Final acceptance remains blocked until DATA_AUDIT.md gate conditions are met."
