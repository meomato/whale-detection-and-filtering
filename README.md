# whale-detection-and-filtering

Whale sound/noise classification with frozen pretrained audio encoders and a
shared downstream classifier.

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
|   |-- train_downstream.py         # shared sound/noise classifier on embeddings
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
|-- data_loading/                   # public dataset download settings
`-- voxaboxen/                      # external Voxaboxen run settings

docs/
|-- BENCHMARK.md                    # benchmark method, tables, commands, notes
|-- DATASETS.md                     # dataset notes
`-- MODELS.md                       # pretrained model and checkpoint notes

reports/
|-- benchmark_figures/              # 5-second benchmark plots
`-- benchmark_1s_figures/           # 1-second benchmark plots

utils/
`-- datasets_downloads/
    |-- download_watkins.py         # download Watkins marine mammal dataset
    |-- download_noaa_onms.py       # sample small subsets from NOAA ONMS / SanctSound
    |-- download_orcasound.py       # download and process Orcasound AWS Open Data
    `-- download_manual_sed.py      # download manual SED dataset from Google Drive

scripts/
|-- download_animal2vec_pretrained.sh
|-- run_animal2vec_pretrained_chunk.sh
`-- run_animal2vec_pretrained_chunks.sh
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

Main plots:

- [5-second figures](reports/benchmark_figures)
- [1-second figures](reports/benchmark_1s_figures)

Full notes:

- [Benchmark report](docs/BENCHMARK.md)
- [Datasets](docs/DATASETS.md)
- [Models and checkpoints](docs/MODELS.md)
