# -*- coding: utf-8 -*-
import sys, time, math
from pathlib import Path
HERE = Path(__file__).resolve()

ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

# === tes imports existants (features) ===
from MetaMatch.src.meta_space.spectral_features import compute_spectral_metrics_for_all
from MetaMatch.src.meta_space.topology_features import compute_topological_metrics, vector_to_point_cloud
from MetaMatch.src.meta_space.classical_distances import compute_classical_distances
from MetaMatch.src.meta_space.syntax_string_features import compute_syntax_string_features

# === embeddings ===
# Utilise Sentence-Transformers par défaut ; accepte aussi des checkpoints HF (bert, distilbert, bart, etc.)
from sentence_transformers import SentenceTransformer
import torch


# ------------------------------------------------------------
# 1) Helpers: chargement & normalisation
# ------------------------------------------------------------
def _read_csv_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    # On essaie ; si séparateur exotique, laisse l’utilisateur adapter
    return pd.read_csv(path)

from pathlib import Path
import json
import pandas as pd

from pathlib import Path
import json
import pandas as pd

def _norm(s):
    return str(s).strip().lower() if s is not None else ""

def load_golden_pairs(golden_path_or_obj, bidirectional=True):
    """
    Retourne un set de tuples normalisés:
    (source_table, source_column, target_table, target_column)
    """
    if isinstance(golden_path_or_obj, (str, Path)):
        with open(golden_path_or_obj, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = golden_path_or_obj  # déjà chargé (list/dict)

    pairs = set()
    if isinstance(data, list):
        for item in data:
            m = (item or {}).get("matches", {})
            st, sc = _norm(m.get("source_table")), _norm(m.get("source_column"))
            tt, tc = _norm(m.get("target_table")), _norm(m.get("target_column"))
            if st and sc and tt and tc:
                pairs.add((st, sc, tt, tc))
                if bidirectional:
                    pairs.add((tt, tc, st, sc))
    elif isinstance(data, dict) and "matches" in data:
        m = data["matches"]
        st, sc = _norm(m.get("source_table")), _norm(m.get("source_column"))
        tt, tc = _norm(m.get("target_table")), _norm(m.get("target_column"))
        if st and sc and tt and tc:
            pairs.add((st, sc, tt, tc))
            if bidirectional:
                pairs.add((tt, tc, st, sc))
    return pairs

def mark_true_match(df: pd.DataFrame, pairs: set,
                    cols=("source_table","source_column","target_table","target_column")):
    st, sc, tt, tc = cols
    # colonnes normalisées pour comparer proprement
    df["_st"] = df[st].map(_norm)
    df["_sc"] = df[sc].map(_norm)
    df["_tt"] = df[tt].map(_norm)
    df["_tc"] = df[tc].map(_norm)

    df["true_match"] = [
        1 if (row._st, row._sc, row._tt, row._tc) in pairs else 0
        for row in df.itertuples(index=False)
    ]
    return df.drop(columns=["_st","_sc","_tt","_tc"])


def _norm(s):  # normalisation simple
    return str(s).strip().lower() if s is not None else ""

def golden_matrix(
    golden_json_path_or_obj,
    source_table: str,
    target_table: str,
    source_attrs: list[str],
    target_attrs: list[str],
) -> pd.DataFrame:
    """
    Retourne une matrice (DataFrame) 0/1 avec lignes=attrs S1, colonnes=attrs S2.
    """
    # charge le golden
    if isinstance(golden_json_path_or_obj, (str, Path)):
        with open(golden_json_path_or_obj, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = golden_json_path_or_obj

    st_req, tt_req = _norm(source_table), _norm(target_table)

    # set des paires (src_col, tgt_col) présentes dans le golden pour ces tables
    pairs = set()
    for item in (data if isinstance(data, list) else [data]):
        m = (item or {}).get("matches", {})
        if _norm(m.get("source_table")) == st_req and _norm(m.get("target_table")) == tt_req:
            sc, tc = _norm(m.get("source_column")), _norm(m.get("target_column"))
            if sc and tc:
                pairs.add((sc, tc))

    # construit la matrice 0/1
    rows = [str(a) for a in source_attrs]
    cols = [str(b) for b in target_attrs]
    M = pd.DataFrame(0, index=rows, columns=cols, dtype=int)

    for r in rows:
        for c in cols:
            if (_norm(r), _norm(c)) in pairs:
                M.at[r, c] = 1
    return M

def _normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    # L2 normalize rows
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return x / norms


# ------------------------------------------------------------
# 2) Construction de texte de colonne → embedding
# ------------------------------------------------------------
def build_column_texts(
    df: pd.DataFrame,
    colnames: list[str] | None = None,
    n_samples_per_col: int = 20,
    dropna: bool = True,
) -> dict[str, str]:
    """
    Pour chaque colonne, on construit un petit "profil" texte :
    - nom de colonne
    - dtype
    - échantillon de valeurs (jusqu'à n_samples_per_col, cast en str)
    """
    if colnames is None:
        colnames = [str(c) for c in df.columns]

    texts = {}
    for col in colnames:
        ser = df[col]
        # dtype + quelques valeurs
        dtype = str(ser.dtype)
        if dropna:
            ser = ser.dropna()

        # échantillon (aléatoire si long)
        if len(ser) > n_samples_per_col:
            sample_vals = ser.sample(n_samples_per_col, random_state=42)
        else:
            sample_vals = ser

        # cast en str + nettoyage léger
        vals = [str(v)[:200] for v in sample_vals.tolist()]
        snippet = " | ".join(vals)

        text = f"column_name: {col}\ndtype: {dtype}\nsample_values: {snippet}"
        texts[str(col)] = text

    return texts


def embed_texts(
    texts_by_key: dict[str, str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str | None = None,
    batch_size: int = 64,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    text -> embeddings ; retourne DataFrame (index = keys, colonnes = dims)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(model_name, device=device)

    keys = list(texts_by_key.keys())
    texts = [texts_by_key[k] for k in keys]

    embs = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start+batch_size]
        E = model.encode(chunk, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False)
        embs.append(E)
    X = np.vstack(embs)

    if normalize:
        X = _normalize(X)

    # colonnes = dim_0 ... dim_{d-1}
    cols = [f"dim_{i}" for i in range(X.shape[1])]
    df_emb = pd.DataFrame(X, index=keys, columns=cols)
    return df_emb


def pick_model_checkpoint(alias: str) -> str:
    """
    Autorise des alias 'bert', 'distilbert', 'bart', etc.,
    sinon renvoie tel quel (checkpoint HF/SBERT).
    """
    a = alias.lower()
    if a in {"bert", "bert-base"}:
        # BERT base (uncased) via Sentence-Transformers wrapper
        return "sentence-transformers/bert-base-nli-mean-tokens"
    if a in {"distilbert", "distilbert-base"}:
        return "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"
    if a in {"bart"}:
        # BART n’est pas typiquement un encodeur de phrase ; on peut passer par "facebook/bart-base" avec pooling,
        # mais SentenceTransformer propose un wrapper universel "all-MiniLM" plus simple.
        # On donne un alias BART → MiniLM (pratique). Sinon, remplace par un modèle HF adapté si tu veux.
        return "sentence-transformers/all-MiniLM-L6-v2"
    if a in {"mini", "minilm"}:
        return "sentence-transformers/all-MiniLM-L6-v2"
    if a in {"bge-base"}:
        return "BAAI/bge-base-en-v1.5"
    if a in {"e5-base"}:
        return "intfloat/e5-base-v2"
    # sinon on suppose que c’est un vrai checkpoint
    return alias


# ------------------------------------------------------------
# 3) compute_distances (corrigée: return après les boucles)
# ------------------------------------------------------------
def compute_distances(
    embedd_ds1: pd.DataFrame,        # lignes = attributs source, colonnes = dims
    embedd_ds2: pd.DataFrame,        # lignes = attributs target, colonnes = dims
    golden_matrix: pd.DataFrame,     # G : lignes = noms source, colonnes = noms target (binaire 0/1)
    Category: str,
    Relation: str,
    Dataset: str,
    Model: str,
    inter_csv_path: str = str(ROOT / "Experimentation" /"tests"/"results_meta_space"/"intermediaire.csv"),
    flush_every: int = 1000
) -> pd.DataFrame:

    results = []

    # Orientation attendue
    attrs1 = embedd_ds1.index.astype(str).tolist()
    attrs2 = embedd_ds2.index.astype(str).tolist()
    mat1 = embedd_ds1.to_numpy(dtype=float)
    mat2 = embedd_ds2.to_numpy(dtype=float)

    # Checks
    if mat1.ndim != 2 or mat2.ndim != 2:
        raise ValueError("embedd_ds1/embedd_ds2 doivent être des matrices 2D (lignes=attributs, colonnes=dimensions).")
    if mat1.shape[1] != mat2.shape[1]:
        raise ValueError(f"Dimensions d'embedding différentes: source={mat1.shape[1]} vs target={mat2.shape[1]}.")

    # Golden
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
            start_time = time.time()

            val = 0.0
            if has_golden:
                try:
                    if a1 in golden_matrix.index and a2 in golden_matrix.columns:
                        val = float(golden_matrix.at[a1, a2])
                    else:
                        # fallback par position si tailles alignées
                        val = float(golden_matrix.iloc[i, j])
                except Exception:
                    val = 0.0

            row = {
                "Attribute1": a1,
                "Attribute2": a2,
                "true_match": val,
                "Category": Category,
                "Relation": Relation,
                "Dataset": Dataset,
                "Model": Model,
            }

            # Features
            row.update(compute_classical_distances(v1, v2))
            row.update(compute_spectral_metrics_for_all(cloud_a1, cloud_a2, mat1, mat2))
            for metric_option in ["euclidean", "cosine", "manhattan"]:
                row.update(compute_topological_metrics(v1, v2, mat1, mat2, metric_option))
            row.update(compute_syntax_string_features(a1, a2))

            row["ExecutionTime"] = time.time() - start_time
            results.append(row)
            cnt += 1

            # flush périodique
            if inter_csv_path and (cnt % flush_every == 0):
                pd.DataFrame(results).to_csv(inter_csv_path, index=False)

    df_result = pd.DataFrame(results)
    if inter_csv_path:
        pd.DataFrame(results).to_csv(inter_csv_path, index=False)
    return df_result


# ------------------------------------------------------------
# 4) Orchestrateur: run_for_dataset (path1, path2, path3 + model + dataset)
# ------------------------------------------------------------
def run_for_dataset(
    dataset_name: str,
    path1: str | Path,
    path2: str | Path,
    path3: str | Path | None,   # golden peut être None
    embedding_model: str,       # "bert", "distilbert", "bart", ou checkpoint complet
    out_dir: str | Path = HERE.parent[3] / "MetaMatch/tests/results_meta_space",
    n_samples_per_col: int = 20,
    category: str = "DefaultCategory",
    relation: str = "src→tgt",
    device: str | None = None,
) -> pd.DataFrame:

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Lecture
    df_src = _read_csv_any(path1)
    df_tgt = _read_csv_any(path2)
    golden = _load_golden(path3) if path3 else pd.DataFrame()
    print(golden)

    # 2) Textes → embeddings
    model_ckpt = pick_model_checkpoint(embedding_model)

    texts_src = build_column_texts(df_src, n_samples_per_col=n_samples_per_col)
    texts_tgt = build_column_texts(df_tgt, n_samples_per_col=n_samples_per_col)

    emb_src = embed_texts(texts_src, model_name=model_ckpt, device=device, normalize=True)
    emb_tgt = embed_texts(texts_tgt, model_name=model_ckpt, device=device, normalize=True)

    # 3) Distances + meta-features
    inter_csv = out_dir / f"inter_{dataset_name}__{relation}__{Path(path1).stem}__{Path(path2).stem}__{embedding_model}.csv"
    df_features = compute_distances(
        embedd_ds1=emb_src,
        embedd_ds2=emb_tgt,
        golden_matrix=golden,
        Category=category,
        Relation=relation,
        Dataset=dataset_name,
        Model=embedding_model,
        inter_csv_path=str(inter_csv),
        flush_every=2000,
    )

    # 4) Sauvegarde finale
    out_csv = out_dir / f"Meta_Space__{dataset_name}__{embedding_model}.csv"
    df_features.to_csv(out_csv, index=False)
    print(f"[OK] Résultats écrits: {out_csv}")
    return df_features


# ------------------------------------------------------------
# 5) Exemple d'appel
# ------------------------------------------------------------
if __name__ == "__main__":
    # Exemple: adapte les chemins
    DATASET = "amazon_google_exp"
    PATH1 = str(ROOT/ "Experimentation"/"Datasets"/"Magellan"/"Unionable"/"amazon_google_exp"/"amazon_google_exp_source.csv")
    PATH2 = str(ROOT/ "Experimentation"/"Datasets"/"Magellan"/"Unionable"/"amazon_google_exp"/"amazon_google_exp_target.csv")
    PATH3 =  str(ROOT/ "Experimentation"/"Datasets"/"Magellan"/"Unionable"/"amazon_google_exp"/"amazon_google_exp_mapping.json")

    _ = run_for_dataset(
        dataset_name=DATASET,
        path1=PATH1,
        path2=PATH2,
        path3=PATH3,                 # JSON pris en charge
        embedding_model="all-MiniLM-L6-v2",  # ou "bert", "distilbert", "BAAI/bge-base-en-v1.5", etc.
        out_dir=HERE.parent[3] / "MetaMatch/tests/results_meta_space",
        n_samples_per_col=20,
        category="Magellan",
        relation="src→tgt",
        device=None,                 # "cuda" si tu veux forcer le GPU
    )

