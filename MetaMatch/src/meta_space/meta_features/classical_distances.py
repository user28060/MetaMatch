import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist

def compute_classical_distances(v1, v2):
    return {
        "Euclidean": np.linalg.norm(v1 - v2),
        "Cosine": cdist([v1], [v2], metric='cosine')[0][0],
        "Pearson": pearsonr(v1, v2)[0],
        "Spearman": spearmanr(v1, v2)[0],
        "Minkowski": cdist([v1], [v2], metric='minkowski', p=3)[0][0],
        "Canberra": cdist([v1], [v2], metric='canberra')[0][0],
        "Chebyshev": cdist([v1], [v2], metric='chebyshev')[0][0],
    }
