param([string]$Gcc = $env:VOLTR_GCC)

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$lzoRoot = Join-Path $projectRoot 'lzo-2.10'
$nativeDir = $projectRoot
$outputDll = Join-Path $nativeDir 'lzo2.dll'
if (-not $Gcc) {
    $command = Get-Command gcc.exe -ErrorAction SilentlyContinue
    if ($command) { $Gcc = $command.Source }
}

if (-not $Gcc -or -not (Test-Path -LiteralPath $Gcc -PathType Leaf)) {
    throw 'MinGW GCC was not found; pass -Gcc <path> or set VOLTR_GCC'
}
if (-not (Test-Path -LiteralPath $lzoRoot -PathType Container)) {
    throw "Copied LZO source folder was not found at $lzoRoot"
}

New-Item -ItemType Directory -Path $nativeDir -Force | Out-Null
$sources = Get-ChildItem -LiteralPath (Join-Path $lzoRoot 'src') -File -Filter '*.c' |
    ForEach-Object { $_.FullName }

& $Gcc -shared -O2 "-I$(Join-Path $lzoRoot 'include')" @sources '-o' $outputDll
if ($LASTEXITCODE -ne 0) {
    throw "GCC failed to build $outputDll"
}
Write-Host "Built $outputDll"
