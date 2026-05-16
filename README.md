# cetaceans-filtering

Cetacean audio filtering and whale sound/noise classification with pretrained
bioacoustic and audio encoders.

The project includes data preparation, embedding extraction, downstream
classification, and benchmark reporting for several pretrained models.

## Project Structure

```text
filtering/
|-- embed/
|   `-- perch_v2_embed.py           # compute Perch embeddings for any audio dataset
|-- benchmark/
|   |-- prepare_sound_noise.py      # Label Studio JSON -> binary sound/noise windows
|   |-- extract_animal2vec.py       # frozen animal2vec embeddings
|   |-- extract_voxaboxen_beats.py  # frozen Voxaboxen BEATs embeddings
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
`-- DATASETS.md                     # dataset and checkpoint notes

reports/
`-- benchmark_figures/              # git-friendly benchmark plots

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

## Current Benchmark

The benchmark evaluates pretrained encoders as frozen feature extractors for a
binary whale sound detection task. A shared downstream classifier is trained on
top of each embedding set.

Class mapping:

- `sound`: target whale sounds
- `noise`: background, silence, artifacts, and non-target audio

Tested models:

- animal2vec pretrained MeerKAT checkpoint
- Perch 2.0
- Voxaboxen BEATs

Label sets:

- `annotations_all`: both annotation exports merged
- `annotations_v1`: first annotation export only
- `annotations_v2`: second annotation export only

Result summary:

- Perch 2.0 achieved the best overall frozen-encoder results.
- Voxaboxen BEATs achieved high recall, but produced more false positives.
- animal2vec showed lower frozen-embedding performance and remains relevant
  for the next fine-tuning stage.

| Model | Labels | F1 sound | Recall sound | PR-AUC | FPR |
|---|---|---:|---:|---:|---:|
| Perch 2.0 | annotations_v2 | 0.888 | 0.881 | 0.970 | 0.200 |
| Voxaboxen BEATs | annotations_v2 | 0.831 | 0.942 | 0.924 | 0.616 |
| animal2vec | annotations_v2 | 0.742 | 0.702 | 0.851 | 0.360 |
| Perch 2.0 | annotations_all | 0.616 | 0.703 | 0.610 | 0.257 |
| Voxaboxen BEATs | annotations_all | 0.591 | 0.682 | 0.648 | 0.278 |
| animal2vec | annotations_all | 0.345 | 0.450 | 0.268 | 0.515 |

![Benchmark comparison](reports/benchmark_figures/00_model_metric_comparison.png)

More details:

- Full report: [docs/BENCHMARK.md](docs/BENCHMARK.md)
- Figures: [reports/benchmark_figures](reports/benchmark_figures)
- Datasets and checkpoints: [docs/DATASETS.md](docs/DATASETS.md)
