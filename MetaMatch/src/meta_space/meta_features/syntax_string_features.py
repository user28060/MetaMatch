# -*- coding: utf-8 -*-
"""
Syntax string features (purement Python, sans dépendances externes).
Calcule un ensemble de distances/similarités entre deux libellés (noms d'attributs).
Toutes les features sont préfixées par 'syn_' pour éviter les collisions.
"""

import math
import unicodedata
import re
from collections import Counter

# --------------------- utils ---------------------

_rx_non_alnum = re.compile(r"[^0-9a-z]+", flags=re.IGNORECASE)

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = _strip_accents(str(s)).lower().strip()
    s = _rx_non_alnum.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s: str):
    s = normalize_text(s)
    return s.split() if s else []

def char_ngrams(s: str, n: int):
    s = normalize_text(s).replace(" ", "")
    if n <= 0 or len(s) < n:
        return []
    return [s[i:i+n] for i in range(len(s) - n + 1)]

def jaccard(a, b):
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union

def dice(a, b):
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    return (2 * inter) / (len(A) + len(B))

def cosine_counts(a, b):
    A, B = Counter(a), Counter(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    # dot
    dot = sum(A[k] * B.get(k, 0) for k in A)
    # norms
    na = math.sqrt(sum(v*v for v in A.values()))
    nb = math.sqrt(sum(v*v for v in B.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)

def common_prefix_len(a: str, b: str):
    a, b = a or "", b or ""
    m = min(len(a), len(b))
    i = 0
    while i < m and a[i] == b[i]:
        i += 1
    return i

def common_suffix_len(a: str, b: str):
    a, b = a or "", b or ""
    m = min(len(a), len(b))
    i = 0
    while i < m and a[-(i+1)] == b[-(i+1)]:
        i += 1
    return i

# --------------------- distances ---------------------

def levenshtein(a: str, b: str):
    a, b = a or "", b or ""
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    # DP ligne par ligne
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        ai = a[i-1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j-1] else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j-1] + 1,    # insertion
                prev[j-1] + cost  # substitute
            )
        prev, curr = curr, prev
    return prev[lb]

def osa_damerau(a: str, b: str):
    """Optimal String Alignment (transpositions adjacentes)."""
    a, b = a or "", b or ""
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    d = [[0]*(lb+1) for _ in range(la+1)]
    for i in range(la+1): d[i][0] = i
    for j in range(lb+1): d[0][j] = j
    for i in range(1, la+1):
        for j in range(1, lb+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            d[i][j] = min(
                d[i-1][j] + 1,
                d[i][j-1] + 1,
                d[i-1][j-1] + cost
            )
            if i > 1 and j > 1 and a[i-1] == b[j-2] and a[i-2] == b[j-1]:
                d[i][j] = min(d[i][j], d[i-2][j-2] + 1)
    return d[la][lb]

def lcs_len(a: str, b: str):
    a, b = a or "", b or ""
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0
    # DP O(min(la,lb)) mémoire
    if la < lb:
        a, b = b, a
        la, lb = lb, la
    prev = [0]*(lb+1)
    for i in range(1, la+1):
        curr = [0]*(lb+1)
        ai = a[i-1]
        for j in range(1, lb+1):
            if ai == b[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = curr
    return prev[lb]

def jaro(a: str, b: str):
    a, b = a or "", b or ""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    match_dist = max(0, (max(la, lb) // 2) - 1)
    a_flags = [False]*la
    b_flags = [False]*lb

    matches = 0
    transpositions = 0

    # count matches
    for i in range(la):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, lb)
        for j in range(start, end):
            if not b_flags[j] and a[i] == b[j]:
                a_flags[i] = b_flags[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0

    # count transpositions
    k = 0
    for i in range(la):
        if a_flags[i]:
            while not b_flags[k]:
                k += 1
            if a[i] != b[k]:
                transpositions += 1
            k += 1
    transpositions //= 2

    return (matches/la + matches/lb + (matches - transpositions)/matches) / 3.0

def jaro_winkler(a: str, b: str, p=0.1, max_l=4):
    ja = jaro(a, b)
    # common prefix length up to max_l
    l = 0
    na, nb = a or "", b or ""
    mx = min(max_l, len(na), len(nb))
    while l < mx and na[l] == nb[l]:
        l += 1
    return ja + l * p * (1 - ja)

# --------------------- feature pack ---------------------

def compute_syntax_string_features(name1: str, name2: str) -> dict:
    """
    Calcule un set de features syntaxiques sur les deux libellés.
    Utilise les versions normalisées (sans accents, minuscules, alphanum).
    """
    raw_a = str(name1 or "")
    raw_b = str(name2 or "")
    a = normalize_text(raw_a)
    b = normalize_text(raw_b)

    # tokens & n-grams
    tok_a, tok_b = tokens(raw_a), tokens(raw_b)  # tokens sur version normalisée déjà
    big_a, big_b = char_ngrams(raw_a, 2), char_ngrams(raw_b, 2)
    tri_a, tri_b = char_ngrams(raw_a, 3), char_ngrams(raw_b, 3)

    lev = levenshtein(a, b)
    osa = osa_damerau(a, b)
    lcs = lcs_len(a, b)
    cp  = common_prefix_len(a, b)
    cs  = common_suffix_len(a, b)

    maxlen = max(len(a), len(b), 1)
    lcs_ratio = lcs / maxlen
    cp_ratio  = cp / maxlen
    cs_ratio  = cs / maxlen

    # Similarités
    jacc_tok   = jaccard(tok_a, tok_b)
    dice_tok   = dice(tok_a, tok_b)
    jac_bi     = jaccard(big_a, big_b)
    jac_tri    = jaccard(tri_a, tri_b)
    cos_bi     = cosine_counts(big_a, big_b)
    cos_tri    = cosine_counts(tri_a, tri_b)
    jaro_s     = jaro(a, b)
    jw_s       = jaro_winkler(a, b)

    # Normalisations de distances -> similarités
    lev_sim = 1.0 - (lev / maxlen)
    osa_sim = 1.0 - (osa / maxlen)

    # flags
    eq_exact    = 1.0 if raw_a == raw_b else 0.0
    eq_casefold = 1.0 if raw_a.casefold() == raw_b.casefold() else 0.0
    a_in_b      = 1.0 if a and a in b else 0.0
    b_in_a      = 1.0 if b and b in a else 0.0

    return {
        "syn_len_a": float(len(a)),
        "syn_len_b": float(len(b)),
        "syn_equal_exact": eq_exact,
        "syn_equal_casefold": eq_casefold,
        "syn_contains_a_in_b": a_in_b,
        "syn_contains_b_in_a": b_in_a,

        "syn_levenshtein": float(lev),
        "syn_levenshtein_sim": lev_sim,
        "syn_damerau_osa": float(osa),
        "syn_damerau_osa_sim": osa_sim,

        "syn_lcs_len": float(lcs),
        "syn_lcs_ratio": lcs_ratio,
        "syn_common_prefix_ratio": cp_ratio,
        "syn_common_suffix_ratio": cs_ratio,

        "syn_jaccard_tokens": jacc_tok,
        "syn_dice_tokens": dice_tok,
        "syn_jaccard_bigrams": jac_bi,
        "syn_jaccard_trigrams": jac_tri,
        "syn_cosine_bigrams": cos_bi,
        "syn_cosine_trigrams": cos_tri,

        "syn_jaro": jaro_s,
        "syn_jaro_winkler": jw_s,
    }
