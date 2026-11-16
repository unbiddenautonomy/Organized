param(
    [Parameter(Mandatory = $true)]
    [string]$RawDataDirectory
)

$ErrorActionPreference = 'Stop'
$raw = (Resolve-Path -LiteralPath $RawDataDirectory).Path
if (-not (Test-Path -LiteralPath $raw -PathType Container)) {
    throw "Raw data directory does not exist: $raw"
}
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$targets = @(
    (Join-Path $repo 'research\v2\data'),
    (Join-Path $repo 'research\volpred\data\raw')
)

foreach ($target in $targets) {
    $existing = Get-Item -LiteralPath $target -Force -ErrorAction SilentlyContinue
    if ($existing) {
        if ($existing.LinkType -eq 'Junction' -and
            [string]$existing.Target -eq $raw) {
            Write-Host "Already linked: $target -> $raw"
            continue
        }
        throw "Target already exists; inspect it before linking: $target"
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
    New-Item -ItemType Junction -Path $target -Target $raw | Out-Null
    Write-Host "Linked: $target -> $raw"
}
