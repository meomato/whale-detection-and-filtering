# whale-detection-and-filtering

Binary whale sound detection with pretrained audio encoders.

## Labeling Rule

All stages use the same binary labeling rule:

- `sound`: any target whale sound, including clicks and vocalizations;
- `noise`: everything outside `sound`, including background, silence,
  artifacts, and non-target sounds.

Overlapping `sound` intervals are merged before window labeling, plotting, and
event-level evaluation.

Main documentation:

- [docs/DATASETS.md](docs/DATASETS.md): environment setup, data layout, sources,
  and download commands.
- [docs/MODELS.md](docs/MODELS.md): model sources, checkpoints, sample rates,
  and wrappers.
- [docs/BENCHMARK.md](docs/BENCHMARK.md): frozen-encoder benchmark, metrics,
  tables, and plots.
- [docs/INFERENCE_REVIEW.md](docs/INFERENCE_REVIEW.md): long-file inference
  review with spectrogram plots.

## Project Structure

```text
filtering/
|-- embed/
|   `-- perch_v2_embed.py           # compute Perch embeddings for any audio dataset
|-- benchmark/
|   |-- prepare_sound_noise.py      # Label Studio JSON -> binary sound/noise windows
|   |-- prepare_windows_from_manifests.py # rebuild windows from manifests
|   |-- extract_animal2vec.py       # frozen animal2vec embeddings
|   |-- extract_voxaboxen_beats.py  # frozen Voxaboxen BEATs embeddings
|   |-- extract_wav2vec2.py         # frozen Wav2Vec2 embeddings
|   |-- merge_embedding_chunks.py   # join chunked embedding files
|   |-- train_downstream.py         # shared sound/noise classifier on embeddings
|   |-- make_file_cv_splits.py      # file-level CV split tables
|   |-- threshold_sweep.py          # validation threshold sweeps
|   |-- summarize_cv_benchmark.py   # CV tables and comparison plots
|   `-- summarize_cv_thresholds.py  # threshold tables
|-- review/
|   |-- prepare_inference_windows.py # long-file windows
|   |-- apply_cv_ensemble.py         # apply CV heads to review audio
|   |-- summarize_inference_review.py # spectrogram overlays and review metrics
|   `-- plot_inference_review.py     # manual annotation spectrograms
|-- watkins/
|   |-- train_classifier.py         # multiclass species classifier
|   `-- classifier/                 # data loading, metrics, reporting, pipeline
|-- sed/
|   |-- convert_annotations.py      # annotations.json -> manifest.csv
|   |-- train_classifier.py         # binary sound/noise classifier
|   `-- classifier/                 # labeling, data loading, pipeline
`-- voxaboxen/
    |-- prepare_dataset.py          # Label Studio JSON -> Voxaboxen/Raven files
    |-- run_training.py             # call external Voxaboxen training
    `-- run_inference.py            # call external Voxaboxen inference

configs/
|-- benchmark/                      # benchmark windowing and classifier settings
|-- perch_embeddings/               # Perch embedding configs
|-- perch_training/                 # Watkins Perch classifier settings
|-- sed_training/                   # SED classifier settings
|-- sound_event_detection/          # annotation conversion settings
|-- data_loading/                   # public dataset download settings
`-- voxaboxen/                      # external Voxaboxen run settings

docs/
|-- BENCHMARK.md                    # benchmark method, tables, commands, notes
|-- DATASETS.md                     # setup, data layout, sources, downloads
|-- INFERENCE_REVIEW.md             # long-file annotation and spectrogram notes
`-- MODELS.md                       # pretrained model and checkpoint notes

reports/
|-- benchmark_context5_hop1_cv/      # CV benchmark tables and main plots
|-- benchmark_context5_hop1_figures/ # per-run benchmark plots
`-- inference_review_context5_hop1/  # long-file spectrogram checks

utils/
`-- datasets_downloads/
    |-- audio_saver.py              # shared audio writing helpers
    |-- manifest_utils.py           # shared manifest helpers
    |-- download_watkins.py         # download Watkins marine mammal dataset
    |-- download_noaa_onms.py       # sample small subsets from NOAA ONMS / SanctSound
    |-- download_orcasound.py       # download and process Orcasound AWS Open Data
    |-- download_pacific_sound.py   # download Pacific Sound samples
    |-- download_onc_hydrophones.py # download ONC hydrophone files
    `-- download_voices_in_the_sea.py # download short reference examples

scripts/
|-- hpc/                            # cluster setup and Slurm scripts
|-- download_animal2vec_pretrained.sh
|-- run_animal2vec_pretrained_chunk.sh
|-- run_animal2vec_pretrained_chunks.sh
`-- run_animal2vec_pretrained_extract.sh
```

## Main Results

The current benchmark uses frozen encoders and the same `sound/noise`
downstream head. All models use 5-second audio context with a 1-second hop.

Current CV results:

| Annotation set | Best by | Model | Window AP | F1 | Precision | Recall | FPR |
|---|---|---|---:|---:|---:|---:|---:|
| all | low FPR | Perch 2.0 | 0.806 | 0.693 | 0.773 | 0.633 | 0.143 |
| all | F1 | Voxaboxen BEATs | 0.801 | 0.696 | 0.750 | 0.672 | 0.172 |
| v2 | Window AP / F1 | Voxaboxen BEATs | 0.873 | 0.751 | 0.808 | 0.705 | 0.200 |
| v2 | close baseline | Perch 2.0 | 0.846 | 0.741 | 0.796 | 0.700 | 0.224 |

Perch is the cleaner frozen baseline on the merged annotations. Voxaboxen is
slightly stronger on the smaller v2 set. Frozen animal2vec is behind Perch and
Voxaboxen, but it remains a candidate for later model-specific fine-tuning.

Next fine-tuning candidates are Perch 2.0, Voxaboxen BEATs, and animal2vec.
Wav2Vec2 is kept only as a general frozen baseline.
