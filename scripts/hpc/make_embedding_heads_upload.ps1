param(
    [string]$Output = "whale_project_hpc_embedding_heads_with_embeddings.zip"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path ".").Path
$Stage = Join-Path $Root "_hpc_embedding_heads_upload_staging"
$Zip = Join-Path $Root $Output

if (Test-Path -LiteralPath $Stage) {
    Remove-Item -LiteralPath $Stage -Recurse -Force
}
if (Test-Path -LiteralPath $Zip) {
    Remove-Item -LiteralPath $Zip -Force
}

$Files = @(
    "filtering\finetune\detector_metrics.py",
    "filtering\finetune\train_embedding_head_detector.py",
    "filtering\benchmark\summarize_cv_benchmark.py",
    "scripts\hpc\embedding_head_common.sh",
    "scripts\hpc\wav2vec2_embedding_heads.sbatch",
    "scripts\hpc\animal2vec_embedding_heads.sbatch",
    "scripts\hpc\perch_embedding_heads.sbatch",
    "scripts\hpc\voxaboxen_embedding_heads.sbatch"
)

$EmbeddingDirs = @(
    "outputs\benchmark_context5_hop1\embeddings\wav2vec2_base_annotations_all",
    "outputs\benchmark_context5_hop1\embeddings\animal2vec_pretrained_meerkat_annotations_all",
    "outputs\benchmark_context5_hop1\embeddings\voxaboxen_beats_annotations_all",
    "outputs\benchmark_context5_hop1\embeddings\perch_v2_all_audio"
)

foreach ($Rel in $Files) {
    $Src = Join-Path $Root $Rel
    $Dst = Join-Path $Stage $Rel
    New-Item -ItemType Directory -Force -Path (Split-Path $Dst -Parent) | Out-Null
    Copy-Item -LiteralPath $Src -Destination $Dst -Force
}

foreach ($Rel in $EmbeddingDirs) {
    $Src = Join-Path $Root $Rel
    if (-not (Test-Path -LiteralPath (Join-Path $Src "embeddings.npy"))) {
        Write-Warning "Missing embeddings.npy: $Rel"
        continue
    }
    if (-not (Test-Path -LiteralPath (Join-Path $Src "manifest.csv"))) {
        Write-Warning "Missing manifest.csv: $Rel"
        continue
    }
    $Dst = Join-Path $Stage $Rel
    New-Item -ItemType Directory -Force -Path $Dst | Out-Null
    Copy-Item -LiteralPath (Join-Path $Src "embeddings.npy") -Destination $Dst -Force
    Copy-Item -LiteralPath (Join-Path $Src "manifest.csv") -Destination $Dst -Force
}

Compress-Archive -Path (Join-Path $Stage "*") -DestinationPath $Zip -Force
Remove-Item -LiteralPath $Stage -Recurse -Force
Get-Item -LiteralPath $Zip
