# whale-detection-and-filtering

Whale sound/noise detection experiments with pretrained audio encoders.

Start here:

- [docs/DATASETS.md](docs/DATASETS.md): environment setup, local data layout,
  dataset sources, and download commands.
- [docs/MODELS.md](docs/MODELS.md): pretrained encoders, checkpoints, sample
  rates, and model-specific notes.
- [docs/BENCHMARK.md](docs/BENCHMARK.md): benchmark setup, commands, metrics,
  tables, figures, and current results.

## Project Structure

```text
filtering/
|-- embed/
|   `-- perch_v2_embed.py           # compute Perch embeddings for any audio dataset
|-- benchmark/
|   |-- prepare_sound_noise.py      # Label Studio JSON -> binary sound/noise windows
|   |-- prepare_windows_from_manifests.py # rebuild windows from saved manifests
|   |-- extract_animal2vec.py       # frozen animal2vec embeddings
|   |-- extract_voxaboxen_beats.py  # frozen Voxaboxen BEATs embeddings
|   |-- extract_wav2vec2.py         # frozen Wav2Vec2 embeddings
|   |-- merge_embedding_chunks.py   # join chunked embedding files
|   |-- train_downstream.py         # shared sound/noise classifier on embeddings
|   |-- add_detection_metrics.py    # add event/FAR style detection metrics
|   |-- summarize_results.py        # summary tables and comparison plots
|   `-- collect_figures.py          # copy plots into reports/benchmark_figures
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
|-- MODELS.md                       # pretrained model and checkpoint notes
`-- GPU_SETUP.md                    # local GPU notes, ignored by git

reports/
|-- benchmark_figures/              # 5-second benchmark plots
`-- benchmark_1s_figures/           # 1-second benchmark plots

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
|-- download_animal2vec_pretrained.sh
|-- run_animal2vec_pretrained_chunk.sh
|-- run_animal2vec_pretrained_chunks.sh
`-- run_animal2vec_pretrained_extract.sh
```

## Main Results

Benchmark here means that pretrained models are used only as feature
extractors. Their weights are frozen, embeddings are extracted from the same
audio windows, and the same simple `sound/noise` classifier is trained on top.

Two runs are reported:

- `5-second`: full-clip windows, used as the coarse baseline.
- `1-second`: shorter windows, used because whale sounds can occupy only part
  of a clip.

Best `annotations_v2` results:

| Benchmark | Best model | mAP/AP | F1 | Precision | Recall | FPR |
|---|---|---:|---:|---:|---:|---:|
| 5-second | Perch 2.0 | 0.970 | 0.888 | 0.895 | 0.881 | 0.200 |
| 1-second | Perch 2.0 | 0.934 | 0.816 | 0.886 | 0.757 | 0.186 |

Voxaboxen BEATs gives high recall, especially on `annotations_v2`, but also
more false positives. animal2vec stays as the main candidate for later full
fine-tuning. Wav2Vec2 works as a general audio baseline.

[Benchmark report](docs/BENCHMARK.md)
