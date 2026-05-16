"""Join chunked embedding exports into one benchmark embedding folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def merge(args: argparse.Namespace) -> None:
    chunk_dirs = sorted(path for path in args.chunks_dir.glob("chunk_*") if path.is_dir())
    if not chunk_dirs:
        raise FileNotFoundError(f"No chunk_* directories found under {args.chunks_dir}")

    arrays: list[np.ndarray] = []
    manifests: list[pd.DataFrame] = []
    metadata: list[dict] = []
    offset = 0
    for chunk_dir in chunk_dirs:
        embeddings_path = chunk_dir / "embeddings.npy"
        manifest_path = chunk_dir / "manifest.csv"
        metadata_path = chunk_dir / "metadata.json"
        if not embeddings_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Incomplete chunk: {chunk_dir}")

        arr = np.load(embeddings_path)
        manifest = pd.read_csv(manifest_path)
        if len(arr) != len(manifest):
            raise ValueError(f"{chunk_dir}: embeddings rows != manifest rows")

        manifest = manifest.copy()
        manifest["row_index"] = np.arange(offset, offset + len(manifest))
        offset += len(manifest)
        arrays.append(arr.astype(np.float32, copy=False))
        manifests.append(manifest)
        if metadata_path.is_file():
            metadata.append(json.loads(metadata_path.read_text(encoding="utf-8")))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    X = np.concatenate(arrays, axis=0)
    table = pd.concat(manifests, ignore_index=True)
    np.save(args.output_dir / "embeddings.npy", X)
    table.to_csv(args.output_dir / "manifest.csv", index=False)

    out_meta = {
        "model_name": args.model_name,
        "chunks_dir": str(args.chunks_dir),
        "n_chunks": len(chunk_dirs),
        "n_windows": int(X.shape[0]),
        "embedding_dim": int(X.shape[1]) if X.ndim == 2 and X.shape[0] else None,
        "chunk_metadata": metadata,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(out_meta, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out_meta.items() if k != "chunk_metadata"}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="animal2vec_pretrained_meerkat")
    return parser.parse_args()


if __name__ == "__main__":
    merge(parse_args())
