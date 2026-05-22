# Whale Detection and Filtering

Binary whale sound detection experiments with pretrained audio encoders,
detector-style evaluation, and long-file inference review.

The binary labeling rule is consistent across the project:

| Label | Meaning |
|---|---|
| `sound` | target whale sound, including clicks and vocalizations |
| `noise` | background, silence, artifacts, non-target sounds, and unmarked audio |

Overlapping `sound` intervals are merged before window labeling, plotting, and
event-level evaluation.

## Main Results

### Frozen Encoder Benchmark

The main benchmark uses 5 s windows, 1 s hop, frozen encoders, and the same
downstream logistic `sound/noise` head. Evaluation uses 5-fold file-level
cross-validation.

| Model | Annotation set | Window AP | Window F1 | Main takeaway |
|---|---|---:|---:|---|
| Voxaboxen BEATs | v2 | 0.873 | 0.751 | best frozen score on the cleaner v2 subset |
| Perch 2.0 | v2 | 0.846 | 0.741 | close specialized bioacoustic baseline |
| Perch 2.0 | all | 0.806 | 0.693 | cleanest merged-set baseline, lower FPR |
| Voxaboxen BEATs | all | 0.801 | 0.696 | similar merged-set performance to Perch |
| animal2vec | all | 0.623 | 0.518 | weaker as a frozen encoder |
| Wav2Vec2 base | all | 0.511 | 0.400 | weakest general speech baseline |

Specialized encoders clearly outperform Wav2Vec2. Cleaner labels also matter:
the best results are on `annotations_v2`.

![Frozen encoder benchmark comparison](reports/benchmark_context5_hop1_cv/00_model_metric_comparison.png)

### HPC Tuning

Two tuning directions were tested:

| Tuning branch | Result |
|---|---|
| Full detector fine-tuning | Voxaboxen BEATs is useful: test event F1@0.5 = `0.446`; Wav2Vec2 is weak; animal2vec was too slow/unstable |
| Frozen embeddings + `linear`/`MLP2` heads | larger heads did not solve event detection; best mean event F1@0.5 was only `0.044` |

The important lesson is that good window ranking is not the same as good
event-level detection. A task-specific detector helped Voxaboxen, but simply
adding a larger head to frozen embeddings did not.

![Voxaboxen full detector metrics](reports/tuning/figures/10_voxaboxen_full_detector_metrics.png)

### Long-File Inference Review

The review set is a 30-minute Orcasound fragment plotted as a continuous
timeline with manual annotations and model scores. At the temporary `0.5`
cutoff, Perch gives the best frozen long-file balance:

| Model | Precision | Recall | F1 | AP | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Perch 2.0 | 0.830 | 0.695 | 0.757 | 0.861 | 0.785 |
| Voxaboxen | 0.883 | 0.364 | 0.515 | 0.806 | 0.684 |
| animal2vec | 0.648 | 0.593 | 0.620 | 0.723 | 0.590 |
| Wav2Vec2 | 0.613 | 0.623 | 0.618 | 0.587 | 0.486 |

The tuning inference review is split into separate linear-head, MLP2-head, and
full fine-tune plots so the timelines remain readable.

![Long-file inference score timeline](reports/inference_review_context5_hop1/orcasound_2020-06-26-SRKW-Lpod_cv_annotations_all/01_inference_scores_timeline.png)

## Documentation

| Document | Purpose |
|---|---|
| [docs/DATASETS.md](docs/DATASETS.md) | setup, data layout, sources, and download commands |
| [docs/MODELS.md](docs/MODELS.md) | pretrained encoders, checkpoints, sample rates, and wrappers |
| [docs/BENCHMARK.md](docs/BENCHMARK.md) | frozen-encoder benchmark method, tables, figures, and commands |
| [docs/TUNING.md](docs/TUNING.md) | HPC fine-tuning and two-head tuning results |
| [docs/INFERENCE_REVIEW.md](docs/INFERENCE_REVIEW.md) | long-file inference review, spectrograms, and overlay plots |
| [docs/HYPOTHESIS.md](docs/HYPOTHESIS.md) | five project hypotheses, tests, and outcomes |

## Repository Structure

```text
.
|-- configs/
|   |-- benchmark/              # windowing and frozen-head benchmark settings
|   |-- data_loading/           # public dataset download settings
|   |-- perch_embeddings/       # Perch extraction configs
|   |-- sed_training/           # earlier binary classifier configs
|   `-- voxaboxen/              # external Voxaboxen wrapper settings
|
|-- docs/
|   |-- DATASETS.md             # setup, data sources, local data layout
|   |-- MODELS.md               # pretrained encoders and checkpoint notes
|   |-- BENCHMARK.md            # frozen-encoder CV benchmark
|   |-- TUNING.md               # HPC full fine-tuning and two-head tuning
|   |-- INFERENCE_REVIEW.md     # 30-minute long-file visual review
|   `-- HYPOTHESIS.md           # main hypothesis, statistical tests, decisions
|
|-- filtering/
|   |-- benchmark/              # Label Studio -> windows, embeddings, CV heads
|   |-- embed/                  # Perch embedding wrapper
|   |-- finetune/               # detector metrics and HPC tuning entrypoints
|   |-- review/                 # long-file windows, inference, spectrogram plots
|   |-- sed/                    # earlier local sound/noise classifier pipeline
|   |-- voxaboxen/              # helpers around external Voxaboxen runs
|   `-- watkins/                # reference species-classification experiments
|
|-- reports/
|   |-- benchmark_context5_hop1_cv/       # main benchmark tables and figures
|   |-- inference_review_context5_hop1/   # long-file timelines and overlays
|   `-- tuning/                         # HPC tuning tables and figures
|
|-- scripts/
|   |-- hpc/                    # Slurm scripts, HPC setup, upload helpers
|   `-- run_animal2vec_*.sh     # animal2vec extraction helpers
|
`-- utils/
    `-- datasets_downloads/     # public audio dataset download utilities

```

## Reproducibility

Create the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
.\.venv\Scripts\python.exe -m pip install -e .
```

Optional model groups:

```powershell
uv sync --group wav2vec2
uv sync --group perch
```

Key report commands are documented in the relevant pages:

- benchmark: [docs/BENCHMARK.md](docs/BENCHMARK.md#commands)
- long-file review: [docs/INFERENCE_REVIEW.md](docs/INFERENCE_REVIEW.md#commands)
- HPC tuning scripts: [scripts/hpc](scripts/hpc)

External model assets are intentionally not committed. Voxaboxen uses an
external Voxaboxen checkout and BEATs weights; animal2vec uses the official
animal2vec/fairseq checkout and pretrained MeerKAT checkpoint.

## Limitations

### Data

- The annotated whale dataset is still small for detector training. The results
  are useful for comparing approaches, but stronger claims would need more
  labeled long recordings from different sites, seasons, hydrophones, call
  types, and background-noise conditions.
- The project uses a binary `sound/noise` target. This is appropriate for
  filtering, but it hides species/call-type differences and does not test
  whether the detector can distinguish whale calls from other biological sound
  sources.
- Noise is treated as one broad class. Vessel noise, rain, silence, clipping,
  other animals, and generic ocean background may require different treatment in
  a production detector.
- Annotation uncertainty is not modeled explicitly. Boundary disagreement,
  missed weak calls, and ambiguous calls can affect both window labels and
  event-level matching.

### Evaluation

- Event-level detection is still the hardest part: window AP can be high while
  event F1 remains low after thresholding and merging neighboring positive
  windows into events.
- The 5 s context / 1 s hop setup is a practical compromise, not proof that this
  is the optimal temporal resolution. Shorter or longer calls may need different
  windows, hop sizes, merge gaps, or post-processing.
- Thresholds were selected on validation data and then fixed for test/review.
  This is the correct protocol, but calibration remains a limitation: a model
  can rank windows well and still choose thresholds that produce poor event
  boundaries.
- The 30-minute long-file inference review is a qualitative and diagnostic
  check, not a full deployment benchmark. It shows how models behave on a
  continuous recording, but more long files are needed for robust operational
  estimates.

### Modeling and Compute

- Frozen embeddings are limited by the pretrained representation. The two-head
  tuning experiments show that adding a larger classifier head does not
  automatically solve event detection.
- Full animal2vec fine-tuning was especially constrained by runtime: one epoch
  was too slow for the available HPC wall-time, where jobs were limited to about
  8 hours.
- The benchmark mostly evaluates models as offline detectors. It does not yet
  measure streaming latency, memory use, real-time throughput, or robustness to
  missing/corrupted audio chunks.

### Deployment

- Large model checkpoints, raw audio, generated embeddings, and raw HPC run
  folders are intentionally outside git. This keeps the repository usable, but
  reproducing every experiment requires recreating local artifacts.
- External dependencies are substantial. Voxaboxen, BEATs, animal2vec/fairseq,
  Perch, PyTorch, and TensorFlow-style environments have different runtime
  assumptions, so environment setup is part of the experimental risk.
- The detector has not been validated on a large multi-site held-out deployment
  corpus. Before operational use, it should be tested on more hydrophones,
  seasons, noise regimes, and whale populations.
