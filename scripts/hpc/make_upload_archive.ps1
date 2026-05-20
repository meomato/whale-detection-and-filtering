param(
    [string]$Archive = "whale_project_hpc.zip"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$archivePath = Join-Path $root $Archive
if (Test-Path $archivePath) {
    Remove-Item -LiteralPath $archivePath -Force
}

$items = @(
    "filtering",
    "configs",
    "scripts",
    "docs",
    "README.md",
    "pyproject.toml",
    "uv.lock"
)

$paths = $items | ForEach-Object { Join-Path $root $_ } | Where-Object { Test-Path $_ }
Compress-Archive -Path $paths -DestinationPath $archivePath -CompressionLevel Optimal
Write-Host "Created $archivePath"
