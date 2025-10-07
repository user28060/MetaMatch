# pipeline.py
from __future__ import annotations
import time
from pathlib import Path
from typing import Optional
import sys

import click
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meta_features.spectral_features import compute_spectral_metrics_for_all
from meta_features.topology_features import compute_topological_metrics, vector_to_point_cloud
from meta_features.classical_distances import compute_classical_distances
from meta_features.syntax_string_features import compute_syntax_string_features

from golden_tools import (
    golden_matrix_s1xs2,
    golden_matrix_s1xs1,
    _non_index_columns,
)
from embed_utils import (
    build_column_texts,
    pick_model_checkpoint,
    embed_texts,
)


def read_csv_any(path: str | Path) -> pd.DataFrame:
    """Read a CSV with a robust delimiter heuristic."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.read_csv(p, sep=None, engine="python")


def compute_distances(
    embedd_ds1: pd.DataFrame,
    embedd_ds2: pd.DataFrame,
    golden_matrix: pd.DataFrame,
    category: str,
    relation: str,
    dataset: str,
    model_name: str,
    inter_csv_path: Optional[Path] = None,
    flush_every: int = 1000,
) -> pd.DataFrame:
    """Compute feature vectors for (S1_attr, S2_attr) pairs and attach true_match from a golden matrix."""
    results = []

    attrs1 = embedd_ds1.index.astype(str).tolist()
    attrs2 = embedd_ds2.index.astype(str).tolist()
    mat1 = embedd_ds1.to_numpy(dtype=float)
    mat2 = embedd_ds2.to_numpy(dtype=float)

    if mat1.ndim != 2 or mat2.ndim != 2:
        raise ValueError("embedd_ds1/embedd_ds2 must be 2D matrices (rows=attributes, cols=dimensions).")
    if mat1.shape[1] != mat2.shape[1]:
        raise ValueError(f"Embedding dimensions differ: source={mat1.shape[1]} vs target={mat2.shape[1]}.")

    has_golden = isinstance(golden_matrix, pd.DataFrame) and not golden_matrix.empty
    if has_golden:
        golden_matrix = golden_matrix.copy()
        golden_matrix.index = golden_matrix.index.astype(str)
        golden_matrix.columns = golden_matrix.columns.astype(str)

    cnt = 0
    for i, a1 in enumerate(attrs1):
        v1 = mat1[i, :]
        cloud_a1 = vector_to_point_cloud(v1)

        for j, a2 in enumerate(attrs2):
            v2 = mat2[j, :]
            cloud_a2 = vector_to_point_cloud(v2)
            t0 = time.time()

            val = 0.0
            if has_golden:
                try:
                    if a1 in golden_matrix.index and a2 in golden_matrix.columns:
                        val = float(golden_matrix.at[a1, a2])
                    else:
                        val = float(golden_matrix.iloc[i, j])
                except Exception:
                    val = 0.0

            row = {
                "Attribute1": a1,
                "Attribute2": a2,
                "true_match": val,
                "Category": category,
                "Relation": relation,
                "Dataset": dataset,
                "Model": model_name,
            }

            row.update(compute_classical_distances(v1, v2))
            row.update(compute_spectral_metrics_for_all(cloud_a1, cloud_a2, mat1, mat2))
            for metric_option in ["euclidean", "cosine", "manhattan"]:
                row.update(compute_topological_metrics(v1, v2, mat1, mat2, metric_option))
            row.update(compute_syntax_string_features(a1, a2))

            row["ExecutionTime"] = time.time() - t0
            results.append(row)
            cnt += 1

            if inter_csv_path and (cnt % flush_every == 0):
                pd.DataFrame(results).to_csv(inter_csv_path, index=False)

    df = pd.DataFrame(results)
    if inter_csv_path:
        df.to_csv(inter_csv_path, index=False)
    return df


@click.command(name="MetaMatch", context_settings={"show_default": True})
@click.option("--dataset", required=True, help="Dataset name for bookkeeping.")
@click.option("--source-csv", "source_csv", required=True, type=click.Path(exists=True, dir_okay=False), help="Path to source CSV (S1).")
@click.option("--target-csv", "target_csv", required=True, type=click.Path(exists=True, dir_okay=False), help="Path to target CSV (S2).")
@click.option("--golden-json", "golden_json", required=False, type=click.Path(exists=True, dir_okay=False), help="Path to golden JSON.")
@click.option("--model", "model_alias", default="all-MiniLM-L6-v2", help="Embedding model alias or HF checkpoint.")
@click.option("--out-dir", "out_dir", default="MetaMatch/tests/results_meta_space", type=click.Path(file_okay=False), help="Output directory.")
def meta_match(
    dataset: str,
    source_csv: str,
    target_csv: str,
    golden_json: Optional[str],
    model_alias: str,
    out_dir: str,
) -> None:
    """Run MetaMatch: read CSVs, embed columns, build golden matrix, compute features, save results."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_src = read_csv_any(source_csv)
    df_tgt = read_csv_any(target_csv)

    src_attrs = _non_index_columns(df_src)
    tgt_attrs = _non_index_columns(df_tgt)

    model_ckpt = pick_model_checkpoint(model_alias)

    texts_src = build_column_texts(df_src, colnames=src_attrs, n_samples_per_col=20)
    texts_tgt = build_column_texts(df_tgt, colnames=tgt_attrs, n_samples_per_col=20)

    emb_src = embed_texts(texts_src, model_name=model_ckpt, device=None, normalize=True)
    emb_tgt = embed_texts(texts_tgt, model_name=model_ckpt, device=None, normalize=True)

    if golden_json:
        G = golden_matrix_s1xs2(golden_json, src_attrs, tgt_attrs)
    else:
        G = pd.DataFrame()

    inter_csv = out_path / f"inter_{dataset}__{Path(source_csv).stem}__{Path(target_csv).stem}__{Path(model_ckpt).name}.csv"
    df_features = compute_distances(
        embedd_ds1=emb_src,
        embedd_ds2=emb_tgt,
        golden_matrix=G,
        category="Magellan",
        relation="Unionable",
        dataset=dataset,
        model_name=model_alias,
        inter_csv_path=inter_csv,
        flush_every=2000,
    )
    out_csv = out_path / f"Meta_Space__{dataset}__{Path(model_ckpt).name}.csv"
    df_features.to_csv(out_csv, index=False)
    click.echo(f"[OK] Saved: {out_csv}")


if __name__ == "__main__":
    meta_match()
