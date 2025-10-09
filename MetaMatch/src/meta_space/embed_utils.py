# embed_utils.py
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def build_column_texts(
    df: pd.DataFrame,
    colnames: List[str] | None = None,
    n_samples_per_col: int = 20,
    dropna: bool = True,
) -> Dict[str, str]:
    """Create a short textual profile per column (name, dtype, value samples)."""
    if colnames is None:
        colnames = [str(c) for c in df.columns]

    texts = {}
    for col in colnames:
        ser = df[col]
        dtype = str(ser.dtype)
        if dropna:
            ser = ser.dropna()
        sample_vals = ser.sample(n_samples_per_col, random_state=42) if len(ser) > n_samples_per_col else ser
        vals = [str(v)[:200] for v in sample_vals.tolist()]
        text = f"column_name: {col}\ndtype: {dtype}\nsample_values: " + " | ".join(vals)
        texts[str(col)] = text
    return texts


def _normalize_rows(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """L2-normalize each row of a 2D array."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return x / norms


def pick_model_checkpoint(alias: str) -> str:
    """Map simple aliases to Sentence-Transformers/HF checkpoints."""
    a = alias.lower()
    if a in {"bert", "bert-base"}:
        return "sentence-transformers/bert-base-nli-mean-tokens"
    if a in {"distilbert", "distilbert-base"}:
        return "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"
    if a in {"bart", "mini", "minilm"}:
        return "sentence-transformers/all-MiniLM-L6-v2"
    if a in {"bge-base"}:
        return "BAAI/bge-base-en-v1.5"
    if a in {"e5-base"}:
        return "intfloat/e5-base-v2"
    return alias


def embed_texts(
    texts_by_key: Dict[str, str],
    model_name: str,
    device: Optional[str] = None,
    batch_size: int = 64,
    normalize: bool = True,
) -> pd.DataFrame:
    """Encode texts into embeddings; returns a DataFrame (index=keys, columns=dimensions)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(model_name, device=device)
    keys = list(texts_by_key.keys())
    texts = [texts_by_key[k] for k in keys]

    embs = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        E = model.encode(
            chunk,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        embs.append(E)
    X = np.vstack(embs) if embs else np.zeros((0, 384), dtype="float32")  # fallback dim

    if normalize and X.size:
        X = _normalize_rows(X)

    cols = [f"dim_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, index=keys, columns=cols)
