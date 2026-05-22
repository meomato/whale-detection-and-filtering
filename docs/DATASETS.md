# Datasets

Run the commands from the repository root. Downloaded audio, annotations, and
checkpoints stay outside git.

## Python Setup

Python 3.11 or newer is required.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
.\.venv\Scripts\python.exe -m pip install -e .
```

Linux or WSL:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -e .
```

Optional dependencies for Wav2Vec2:

```powershell
.\.venv\Scripts\python.exe -m pip install "torch>=2.5" "transformers>=4.51"
```

Optional Perch/Hoplite install:

```powershell
.\.venv\Scripts\python.exe -m pip install "perch-hoplite[tf] @ git+https://github.com/google-research/perch-hoplite"
```

The same setup can be done with `uv`:

```powershell
uv sync
uv sync --group wav2vec2
uv sync --group perch
```

On Windows, the regular project scripts work from `.venv`. For GPU-heavy Perch
or TensorFlow runs, WSL is recommended.

## Data Folder

Data is not committed to git. Keep local audio and annotations under `data/`.

Expected folders:

```text
data/
|-- benchmark/
|   |-- audio/
|   `-- annotations/
|       |-- annotations_v1.json
|       `-- annotations_v2.json
|-- orcasound/
|-- noaa_onms/
|-- pacific_sound/
|-- watkins/
|-- voices_in_the_sea/
`-- voxaboxen/
    |-- audio/
    `-- annotations/
```

The default download path is configured here:

```text
configs/data_loading/data_loading.yaml
```

By default it points to:

```text
data
```

In the current local experiment artifacts, the raw WAV corpus summary contains
60 files and 16.65 hours of audio. The final merged supervised benchmark is the
annotated subset used in `outputs/benchmark_context5_hop1/annotations_all`: 55
files, 13.67 hours, 5,627 sound intervals, and 49,254 labeled windows.

## Orcasound

Source registry:

https://registry.opendata.aws/orcasound/

Useful prefixes:

```text
2020-06-26-SRKW-Lpod/
2021_9_12_OS_YearsBestVocalPassby/
2019-Orcasound-examples/
humpbacks/
2018-sperm-whale-Yukusam/
2017_8_VesselsAndWavS/
```

Download about 30 minutes from one prefix:

```powershell
python utils\datasets_downloads\download_orcasound.py `
  data_loading.raw_datasets_path=data `
  data_loading.raw_segment_duration=-1 `
  data_loading.sources.orcasound.selected_prefixes=[2020-06-26-SRKW-Lpod/] `
  data_loading.sources.orcasound.target_hours_total=0.5 `
  data_loading.sources.orcasound.assume_minutes_per_file=1 `
  data_loading.sources.orcasound.max_files_per_source=60 `
  data_loading.sources.orcasound.only_new_files=true
```

Output:

```text
data/orcasound/<source_name>/audio/
data/orcasound/<source_name>/manifest.jsonl
```

Use Orcasound for long-file inference checks and for additional positive
examples.

## NOAA ONMS / SanctSound

NOAA/SanctSound is useful for hard negatives: ocean background, vessels, and
recordings from different conditions.

Citation:

https://doi.org/10.25921/saca-sp25

Small download:

```powershell
python utils\datasets_downloads\download_noaa_onms.py `
  data_loading.raw_datasets_path=data `
  data_loading.sources.noaa.only_new_files=true `
  data_loading.sources.noaa.hours_per_deployment=0.5 `
  data_loading.raw_segment_duration=-1
```

One file from each configured deployment:

```powershell
python utils\datasets_downloads\download_noaa_onms.py `
  data_loading.raw_datasets_path=data `
  data_loading.sources.noaa.only_new_files=true `
  data_loading.sources.noaa.max_files_per_deployment=1 `
  data_loading.raw_segment_duration=-1
```

Output:

```text
data/noaa_onms/
```

The deployment list is in:

```text
configs/data_loading/data_loading.yaml
```

## Pacific Sound

Pacific Sound is useful as an extra background stress test. It is mainly for
checking false alarms on ocean noise.

Example:

```powershell
python utils\datasets_downloads\download_pacific_sound.py `
  data_loading.raw_datasets_path=data `
  data_loading.sources.pacific_sound.tier=16khz `
  data_loading.sources.pacific_sound.years=[2023] `
  data_loading.sources.pacific_sound.months=[11] `
  data_loading.raw_segment_duration=-1
```

Output:

```text
data/pacific_sound/
```

## Watkins

Watkins is a marine mammal clip dataset. It is useful for reference examples and
species-level experiments, but it is not the main long-file detection dataset.

```powershell
python utils\datasets_downloads\download_watkins.py data_loading.raw_datasets_path=data
```

Output:

```text
data/watkins/
```

## Voices in the Sea

Voices in the Sea contains short example sounds. It is useful for quick audio
checks and small reference examples.

```powershell
python utils\datasets_downloads\download_voices_in_the_sea.py data_loading.raw_datasets_path=data
```

Output:

```text
data/voices_in_the_sea/
```

## Voxaboxen Input Data

Default Voxaboxen input paths:

```text
data/voxaboxen/audio/
data/voxaboxen/annotations/annotations.json
configs/voxaboxen/voxaboxen.yaml
```

Prepare data:

```powershell
python filtering/voxaboxen/prepare_dataset.py
```

Run a named dataset:

```powershell
python filtering/voxaboxen/prepare_dataset.py `
  voxaboxen.dataset_name=my_dataset `
  voxaboxen.audio_dir="data/my_dataset/audio" `
  voxaboxen.annotations_json="data/my_dataset/annotations.json"
```
