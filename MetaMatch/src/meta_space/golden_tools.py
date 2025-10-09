# golden_tools.py
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import numpy as np
import pandas as pd


def _norm(s: object) -> str:
    """Lower/strip normalization for robust comparisons."""
    return str(s).strip().lower() if s is not None else ""


def _non_index_columns(df: pd.DataFrame) -> list[str]:
    """Return column names excluding common dumped index columns."""
    cols = [str(c) for c in df.columns]
    drop_like = {"unnamed: 0", "index"}
    return [c for c in cols if _norm(c) not in drop_like]


def load_golden_as_df(
    golden: Union[str, Path, list, dict, pd.DataFrame]
) -> pd.DataFrame:
    """Load a “matches” mapping into a flattened DataFrame.

    Accepts a JSON path, already-loaded list/dict, or a DataFrame.
    The output contains columns like 'matches.source_column' and 'matches.target_column' if present.
    """
    if isinstance(golden, (str, Path)):
        with open(golden, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.json_normalize(data)
    if isinstance(golden, (list, dict)):
        return pd.json_normalize(golden)
    if isinstance(golden, pd.DataFrame):
        return golden.copy()
    raise TypeError("golden must be a JSON path, list/dict, or DataFraame.")


def _pick_col(df: pd.DataFrame, key: str, *aliases: str) -> str | None:
    """Pick a column by exact name or by '...<dot>suffix' fallback and aliases."""
    for c in df.columns:
        cl = c.lower()
        if cl == key or cl.endswith(f".{key}"):
            return c
    for a in aliases:
        for c in df.columns:
            cl = c.lower()
            if cl == a or cl.endswith(f".{a}"):
                return c
    return None


def golden_matrix_s1xs2(
    golden_map: Union[str, Path, list, dict, pd.DataFrame],
    src_attrs: Iterable[str],
    tgt_attrs: Iterable[str],
) -> pd.DataFrame:
    """Build a 0/1 matrix (rows=S1 attributes, cols=S2 attributes) from a golden 'matches' mapping."""
    src_attrs = [str(x) for x in src_attrs]
    tgt_attrs = [str(x) for x in tgt_attrs]

    map_df = load_golden_as_df(golden_map)
    if map_df.empty:
        return pd.DataFrame(np.zeros((len(src_attrs), len(tgt_attrs)), dtype=int),
                            index=src_attrs, columns=tgt_attrs)

    c_src = _pick_col(map_df, "source_column", "source", "src", "left", "s", "source_attr", "source_name")
    c_tgt = _pick_col(map_df, "target_column", "target", "tgt", "right", "t", "target_attr", "target_name")
    if c_src is None or c_tgt is None:
        return pd.DataFrame(np.zeros((len(src_attrs), len(tgt_attrs)), dtype=int),
                            index=src_attrs, columns=tgt_attrs)

    src_pos = {s.strip(): i for i, s in enumerate(src_attrs)}
    tgt_pos = {t.strip(): j for j, t in enumerate(tgt_attrs)}
    rx_col = re.compile(r"^col[_-]?(\d+)$", re.IGNORECASE)

    def _idx(label: object, pos: dict[str, int], limit: int) -> int | None:
        s = str(label).strip()
        if s in pos:
            return pos[s]
        m = rx_col.match(s)
        if m:
            k = int(m.group(1))
            if 0 <= k < limit:
                return k
        return None

    M = np.zeros((len(src_attrs), len(tgt_attrs)), dtype=int)
    miss = 0

    # >>> key change: iterate by position, not attribute names
    for sc, tc in map_df[[c_src, c_tgt]].itertuples(index=False, name=None):
        i = _idx(sc, src_pos, len(src_attrs))
        j = _idx(tc, tgt_pos, len(tgt_attrs))
        if i is not None and j is not None:
            M[i, j] = 1
        else:
            miss += 1
    if miss:
        print(f"[golden] {miss} pair(s) ignored (labels absent/typos).")

    return pd.DataFrame(M, index=src_attrs, columns=tgt_attrs)


def golden_matrix_s1xs1(
    golden_map: Union[str, Path, list, dict, pd.DataFrame],
    src_attrs: Iterable[str],
    tgt_attrs: Iterable[str],
) -> pd.DataFrame:
    """Build a 0/1 matrix (rows=S1 attributes, cols=S1 attributes). Cell (a,b)=1 iff (a in S1 -> b in S2) exists in golden."""
    src_attrs = [str(x) for x in src_attrs]
    tgt_set = {_norm(x) for x in tgt_attrs}

    map_df = load_golden_as_df(golden_map)
    if map_df.empty:
        return pd.DataFrame(np.zeros((len(src_attrs), len(src_attrs)), dtype=int),
                            index=src_attrs, columns=src_attrs)

    c_src = _pick_col(map_df, "source_column", "source", "src", "left", "s", "source_attr", "source_name")
    c_tgt = _pick_col(map_df, "target_column", "target", "tgt", "right", "t", "target_attr", "target_name")
    if c_src is None or c_tgt is None:
        return pd.DataFrame(np.zeros((len(src_attrs), len(src_attrs)), dtype=int),
                            index=src_attrs, columns=src_attrs)

    pairs = set()
    for r in map_df.itertuples(index=False):
        sc = _norm(getattr(r, c_src))
        tc = _norm(getattr(r, c_tgt))
        if sc and tc:
            pairs.add((sc, tc))

    M = pd.DataFrame(0, index=src_attrs, columns=src_attrs, dtype=int)
    for r in src_attrs:
        rn = _norm(r)
        for c in src_attrs:
            cn = _norm(c)
            if cn in tgt_set and (rn, cn) in pairs:
                M.at[r, c] = 1
    return M
