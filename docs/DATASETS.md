# Datasets And Checkpoints

This file keeps the longer dataset and checkpoint notes out of the main README.

## Python Setup

```bash
uv sync
uv sync --group perch
```

All main scripts use Hydra. Config values can be overridden inline:

```bash
uv run python path/to/script.py section.key=value
```

Most defaults live under `configs/`.

## Watkins Dataset

```bash
uv run python utils/datasets_downloads/download_watkins.py
uv run python filtering/embed/perch_v2_embed.py
uv run python filtering/watkins/train_classifier.py
```

## Manual SED Dataset

```bash
uv run python utils/datasets_downloads/download_manual_sed.py
uv run python filtering/sed/convert_annotations.py
uv run python filtering/embed/perch_v2_embed.py
uv run python filtering/sed/train_classifier.py
```

Window label rule:

- overlap with a `sound` event by at least `min_overlap_s`: `sound`
- otherwise: `noise`
- artifact windows can be kept as `noise` or excluded by config

## NOAA ONMS / SanctSound

Deployment list is configured in `configs/data_loading/data_loading.yaml`.

Small update run:

```powershell
uv run python utils/datasets_downloads/download_noaa_onms.py data_loading.sources.noaa.only_new_files=true data_loading.sources.noaa.hours_per_deployment=1.34 data_loading.raw_segment_duration=-1
```

One file from each deployment:

```powershell
uv run python utils/datasets_downloads/download_noaa_onms.py data_loading.sources.noaa.only_new_files=true data_loading.sources.noaa.max_files_per_deployment=1 data_loading.raw_segment_duration=-1
```

Chunk output into 10-second WAV files:

```powershell
uv run python utils/datasets_downloads/download_noaa_onms.py data_loading.sources.noaa.only_new_files=true data_loading.sources.noaa.hours_per_deployment=1.34 data_loading.raw_segment_duration=10
```

Output paths:

- original files: `data/noaa_onms/downloads/...`
- audio for labeling/training: `data/noaa_onms/audio/...`

Please cite NOAA SanctSound/ONMS data according to deployment metadata:
https://doi.org/10.25921/saca-sp25

## Orcasound

Full run:

```powershell
uv run python utils/datasets_downloads/download_orcasound.py data_loading.sources.orcasound.max_files_per_source=null data_loading.sources.orcasound.target_hours_total=null
```

Small run, about 5 hours total:

```powershell
uv run python utils/datasets_downloads/download_orcasound.py data_loading.sources.orcasound.target_hours_total=5 data_loading.sources.orcasound.duration_min_minutes=4 data_loading.sources.orcasound.duration_max_minutes=6 data_loading.sources.orcasound.assume_minutes_per_file=5 data_loading.sources.orcasound.max_files_per_source=null
```

Output paths:

- original files: `data/orcasound/downloads/...`
- audio for labeling/training: `data/orcasound/audio/...`

Source registry: https://registry.opendata.aws/orcasound/

## Voxaboxen

Voxaboxen is used as an external checkout, not copied into this repository.
The default config expects it next to this project as `../voxaboxen`.

Input defaults:

- audio: `data/voxaboxen/audio/`
- Label Studio JSON: `data/voxaboxen/annotations/annotations.json`
- config: `configs/voxaboxen/voxaboxen.yaml`

```powershell
uv run python filtering/voxaboxen/prepare_dataset.py
uv run python filtering/voxaboxen/run_training.py
uv run python filtering/voxaboxen/run_inference.py
```

Run a named dataset without editing YAML:

```powershell
uv run python filtering/voxaboxen/prepare_dataset.py voxaboxen.dataset_name=my_dataset voxaboxen.audio_dir="data/my_dataset/audio" voxaboxen.annotations_json="data/my_dataset/annotations.json"
```

Longer training run:

```powershell
uv run python filtering/voxaboxen/run_training.py voxaboxen.n_epochs=8 voxaboxen.experiment_name=beats_binary_8ep
```

## animal2vec Checkpoint

The benchmark uses the official pretrained MeerKAT checkpoint:

```text
animal2vec_large_pretrained_MeerKAT_240507.pt
```

Source:

- Edmond dataset: https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.ETPUKU
- DOI: `10.17617/3.ETPUKU`
- datafile id used by the helper script: `253220`

Expected checkpoint location inside the external animal2vec checkout:

```text
<animal2vec_repo>/checkpoints/animal2vec_large_pretrained_MeerKAT_240507.pt
```

Download helper for WSL:

```bash
bash scripts/download_animal2vec_pretrained.sh
```
