# Whale Sound/Noise Benchmark

This document describes the first benchmark stage for the whale sound/noise
classification task. Pretrained audio and bioacoustic encoders were evaluated
as frozen feature extractors. For each encoder, embeddings were extracted from
the same audio windows and a shared downstream classifier was trained for
binary classification.

The benchmark is used to select the most suitable model family for the next
stage of full fine-tuning.

## Classification Setup

The task is binary classification with two target classes:

- `sound`: target whale acoustic events
- `noise`: background, silence, artifacts, and non-target audio

All annotated whale sound types were mapped to `sound`. Overlapping annotated
events were merged before assigning labels to fixed-length windows.

## Data Setup

Input files:

- Audio: external WAV dataset
- Annotation v1: Label Studio JSON export
- Annotation v2: Label Studio JSON export

Prepared datasets:

- `annotations_all`: v1 + v2 merged
- `annotations_v1`: only the first annotation file
- `annotations_v2`: only the second annotation file

Windowing:

- window size: `5.0 s`
- hop size: `5.0 s`
- label rule: a window is `sound` if it overlaps any merged sound interval by
  at least `0.1 s`; otherwise it is `noise`

Corpus snapshot:

| Item | Value |
|---|---:|
| WAV files | 60 |
| Total duration | about 16.65 h |
| Channels | mono |
| Source sample rates | 40 kHz, 44.1 kHz, 60.6 kHz, 121.2 kHz, 192 kHz |

For `annotations_v2`, the requested fixed file-level split was used:

- Train IDs: `1, 10, 13, 15, 16, 17, 19, 20, 23, 25, 26, 33, 46, 57, 60`
- Test IDs: `9, 11, 12, 14, 18, 21, 22, 24, 28, 32`

The split is file/session-level, not random window-level, to avoid leakage.

## Models Tested

| Model | Why it was tested | Input rate | Window | Embedding dim | Device |
|---|---|---:|---:|---:|---|
| Perch 2.0 | strong modern bioacoustic baseline | 32 kHz | 5 s | 1536 | GPU via WSL/TensorFlow |
| Voxaboxen BEATs | acoustic pipeline baseline | 16 kHz | 5 s | 768 | GPU/PyTorch |
| animal2vec MeerKAT pretrained | main candidate for later fine-tuning | 8 kHz | 5 s | 1024 | GPU via WSL/PyTorch |
| Wav2Vec2 base | general audio/speech baseline | 16 kHz | 5 s | 768 | GPU/PyTorch |

The animal2vec result below uses the official pretrained checkpoint:

`animal2vec_large_pretrained_MeerKAT_240507.pt`

Checkpoint check:

- size: `5,024,620,014` bytes
- md5: `c0ae0cb16afd0501f00a5955fb6482ed`
- source: Edmond DOI `10.17617/3.ETPUKU`, datafile `253220`

## How It Was Run

Each encoder exported:

```text
embeddings.npy
manifest.csv
```

Then the same downstream classifier was trained for each model and annotation
set:

- classifier: logistic head via `SGDClassifier(loss="log_loss")`
- encoder: frozen
- max epochs: `20`
- early stopping: validation average precision, `patience=3`
- saved metrics: per epoch + final test metrics

General command:

```powershell
python filtering/benchmark/train_downstream.py `
  --model-name MODEL_NAME `
  --scenario annotations_all `
  --embeddings PATH\embeddings.npy `
  --embedding-manifest PATH\manifest.csv `
  --annotations outputs\benchmark\annotations_all\annotations_manifest.csv `
  --splits outputs\benchmark\annotations_all\splits.csv `
  --output-dir outputs\benchmark\runs\MODEL_NAME\annotations_all `
  --epochs 20 `
  --patience 3 `
  --min-sound-overlap-s 0.1
```

The command was repeated for:

- `annotations_all`
- `annotations_v1`
- `annotations_v2`

Summary and plots:

```powershell
python filtering/benchmark/summarize_results.py --root outputs\benchmark --output-dir outputs\benchmark\report
python filtering\benchmark\collect_figures.py --output-dir reports\benchmark_figures
```

## Figures

The git-friendly figure folder is:

`reports/benchmark_figures/`

Main plot:

![Main model comparison](../reports/benchmark_figures/00_model_metric_comparison.png)

The comparison plot includes Perch 2.0, Voxaboxen BEATs, animal2vec
pretrained, and Wav2Vec2 base across all three annotation sets.

Useful per-run plots are named like:

```text
<model>__<annotation_set>__<plot_type>.png
```

Examples:

- `perch_v2__annotations_v2__pr.png`
- `voxaboxen_beats__annotations_v2__confusion.png`
- `animal2vec_pretrained_meerkat__annotations_v2__epochs.png`
- `wav2vec2_base__annotations_v2__roc.png`

Plot types:

- `epochs`: validation F1 / precision / recall by epoch
- `roc`: ROC curve
- `pr`: precision-recall curve
- `confusion`: confusion matrix

## Results

Full CSV:

`outputs/benchmark/report/summary.csv`

Main table:

| Model | Annotation set | Best epoch | Epochs done | Precision | Recall | F1 | PR-AUC | FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Perch 2.0 | all | 1 | 4 | 0.5489 | 0.7025 | 0.6163 | 0.6096 | 0.2569 |
| Voxaboxen BEATs | all | 4 | 9 | 0.5216 | 0.6823 | 0.5912 | 0.6483 | 0.2779 |
| animal2vec pretrained | all | 1 | 4 | 0.2797 | 0.4503 | 0.3450 | 0.2685 | 0.5152 |
| Wav2Vec2 base | all | 9 | 12 | 0.2171 | 0.3558 | 0.2697 | 0.2745 | 0.5697 |
| Perch 2.0 | v1 | 1 | 4 | 0.4004 | 0.5774 | 0.4729 | 0.3825 | 0.2591 |
| Voxaboxen BEATs | v1 | 1 | 6 | 0.3776 | 0.6938 | 0.4890 | 0.6180 | 0.3416 |
| animal2vec pretrained | v1 | 1 | 4 | 0.2406 | 0.5391 | 0.3327 | 0.2363 | 0.5083 |
| Wav2Vec2 base | v1 | 10 | 13 | 0.1484 | 0.3604 | 0.2102 | 0.1850 | 0.6179 |
| Perch 2.0 | v2 | 1 | 4 | 0.8946 | 0.8810 | 0.8878 | 0.9705 | 0.2000 |
| Voxaboxen BEATs | v2 | 1 | 6 | 0.7431 | 0.9424 | 0.8309 | 0.9236 | 0.6161 |
| animal2vec pretrained | v2 | 2 | 5 | 0.7865 | 0.7018 | 0.7417 | 0.8510 | 0.3602 |
| Wav2Vec2 base | v2 | 10 | 13 | 0.7294 | 0.5539 | 0.6296 | 0.7740 | 0.3886 |

## Result Interpretation

### Perch 2.0

Perch 2.0 achieved the best frozen-encoder performance in this benchmark.

It has:

- best F1 on `annotations_all`
- best F1 on `annotations_v2`
- lowest false positive rate among the strong models
- best PR-AUC on `annotations_v2`

These results make Perch 2.0 the strongest baseline for the binary
`sound/noise` task at the frozen-encoder stage.

### Voxaboxen BEATs

Voxaboxen BEATs showed strong recall-oriented behavior.

It has:

- very high recall on `annotations_v2`: `0.9424`
- strong PR-AUC on `annotations_v1` and `annotations_v2`
- higher FPR than Perch, especially on `annotations_v2`

This indicates a higher-sensitivity detector with a larger number of false
positive predictions.

### animal2vec pretrained

animal2vec with official MeerKAT pretrained weights did not achieve the best
frozen-encoder performance in this benchmark.

It is reasonable on `annotations_v2`, but weak on `annotations_all` and
`annotations_v1`. The main issue is that the frozen features do not separate
sound/noise well enough on this whale dataset.

The result does not exclude animal2vec from further experiments. The model
should be evaluated next with full fine-tuning, because it remains the main
candidate for a task-specific bioacoustic classifier.

### Wav2Vec2 base

Wav2Vec2 base was included as a general audio/speech baseline.

It has:

- lower F1 than all specialized models on `annotations_v2`
- low performance on `annotations_all` and `annotations_v1`
- moderate PR-AUC on `annotations_v2`, but lower recall than the bioacoustic
  baselines

These results indicate that general speech-oriented representations are less
suitable for this whale sound/noise task than specialized bioacoustic
representations.
