"""Semantic text embeddings for Persian Twitter bot detection.

This is an additive module. It does not modify features.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384


def _text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().strip("\"'")


def build_semantic_texts(df: pd.DataFrame) -> list[str]:
    """Combine each account's profile description and sampled status text."""
    descriptions = df["description"] if "description" in df.columns else pd.Series(
        "", index=df.index)
    statuses = df["status"] if "status" in df.columns else pd.Series(
        "", index=df.index)

    texts = []
    for description, status in zip(descriptions, statuses):
        description = _text(description)
        status = _text(status)
        # Explicit separators help the encoder distinguish profile and tweet text.
        texts.append(f"Profile: {description}\nTweet: {status}".strip())
    return texts


def _cache_key(texts: Iterable[str], model_name: str) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(b"\0")
    for item in texts:
        h.update(item.encode("utf-8", errors="replace"))
        h.update(b"\0")
    return h.hexdigest()[:24]


def engineer_semantic_features(
    df: pd.DataFrame,
    cache_dir: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
) -> pd.DataFrame:
    """Return one 384-dimensional embedding per row, with disk caching.

    The pretrained encoder is frozen; no model is fitted on nested_cv labels.
    Cache keys include model name and ordered input texts.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Sentence Transformers is required. Install it with: "
            "pip install sentence-transformers"
        ) from exc

    texts = build_semantic_texts(df)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(texts, model_name)
    cache_path = cache_dir / f"embeddings_{key}.npz"
    metadata_path = cache_dir / f"embeddings_{key}.json"

    embeddings = None
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as saved:
                cached = saved["embeddings"]
                if cached.shape == (len(texts), EMBEDDING_DIM):
                    embeddings = cached.astype(np.float32, copy=False)
                    print(f"Loaded cached semantic embeddings: {cache_path}")
        except (OSError, ValueError, KeyError):
            embeddings = None

    if embeddings is None:
        print(f"Loading semantic model: {model_name}")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        ).astype(np.float32, copy=False)
        if embeddings.shape != (len(texts), EMBEDDING_DIM):
            raise ValueError(
                f"Expected embeddings with shape ({len(texts)}, {EMBEDDING_DIM}), "
                f"got {embeddings.shape}. Check the selected model."
            )
        np.savez_compressed(cache_path, embeddings=embeddings)
        metadata_path.write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "rows": len(texts),
                    "dimension": int(embeddings.shape[1]),
                    "normalized": normalize_embeddings,
                    "cache_key": key,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved semantic embeddings: {cache_path}")

    columns = [f"semantic_{i:03d}" for i in range(EMBEDDING_DIM)]
    return pd.DataFrame(embeddings, index=df.index, columns=columns)
