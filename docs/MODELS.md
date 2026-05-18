# Models

Pretrained encoders used in the frozen benchmark.

## Perch 2.0

Perch 2.0 runs through the project embedding wrapper:

```text
filtering/embed/perch_v2_embed.py
```

5-second benchmark settings:

- sample rate: `32 kHz`
- window size: `5 s`
- embedding dimension: `1536`
- encoder state: frozen

The 1-second benchmark uses `1 s` shards with `1 s` hop.

Source:

- Perch / Hoplite repository: https://github.com/google-research/perch-hoplite

## Voxaboxen BEATs

Voxaboxen stays as an external checkout and is not copied into this repository.

Benchmark wrapper:

```text
filtering/benchmark/extract_voxaboxen_beats.py
```

5-second benchmark settings:

- sample rate: `16 kHz`
- window size: `5 s`
- embedding dimension: `768`
- encoder state: frozen

The 1-second benchmark uses `1 s` windows with `1 s` hop.

Expected checkpoint:

```text
BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
```

## animal2vec

Official pretrained MeerKAT checkpoint:

```text
animal2vec_large_pretrained_MeerKAT_240507.pt
```

Benchmark wrapper:

```text
filtering/benchmark/extract_animal2vec.py
```

5-second benchmark settings:

- sample rate: `8 kHz`
- window size: `5 s`
- embedding dimension: `1024`
- encoder state: frozen

The 1-second benchmark uses `1 s` windows with `1 s` hop.

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

## Wav2Vec2 Base

Wav2Vec2 base is the general audio/speech baseline.

Benchmark wrapper:

```text
filtering/benchmark/extract_wav2vec2.py
```

5-second benchmark settings:

- model: `facebook/wav2vec2-base`
- sample rate: `16 kHz`
- window size: `5 s`
- embedding dimension: `768`
- embedding rule: mean pooling over the final hidden state
- encoder state: frozen

The 1-second benchmark uses `1 s` windows with `1 s` hop.

Local model files stay outside git:

```text
models/wav2vec2_base/
```

Source:

- Hugging Face model card: https://huggingface.co/facebook/wav2vec2-base
- Hugging Face Transformers Wav2Vec2 documentation:
  https://huggingface.co/docs/transformers/en/model_doc/wav2vec2
