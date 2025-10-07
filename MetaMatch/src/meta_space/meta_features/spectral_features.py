import numpy as np
from scipy.linalg import svd

def alpha_reQ(matrix):
    U, S, _ = svd(matrix, full_matrices=False)
    log_indices = np.log(np.arange(1, len(S) + 1))
    log_singular_values = np.log(S)
    slope, _ = np.polyfit(log_indices, log_singular_values, 1)
    return -slope

def NESum(matrix):
    C = np.cov(matrix, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(C)
    eigenvalues = sorted(eigenvalues, reverse=True)
    lambda_0 = eigenvalues[0] if eigenvalues[0] > 0 else 1e-6
    return np.sum(eigenvalues / lambda_0)

def rank_me(matrix):
    _, S, _ = svd(matrix, full_matrices=False)
    S_normalized = S / np.sum(S)
    return np.exp(-np.sum(S_normalized * np.log(S_normalized + 1e-12)))

def stable_rank(matrix):
    frobenius_norm = np.linalg.norm(matrix, 'fro')
    spectral_norm = np.linalg.norm(matrix, 2)
    return (frobenius_norm ** 2) / (spectral_norm ** 2)

def self_cluster(matrix):
    n, d = matrix.shape
    dot_product = matrix @ matrix.T
    frobenius_norm = np.linalg.norm(dot_product, 'fro')
    return (d * frobenius_norm - n * (d + n - 1)) / ((d - 1) * (n - 1) * n)

def compute_spectral_metrics_for_all(cloud_a1, cloud_a2, mat_A1, mat_A2):


    metrics = {}

    for name, func in {
        "alpha_ReQ": alpha_reQ,
        "NESum": NESum,
        "RankMe": rank_me,
        "StableRank": stable_rank,
        "SelfCluster": self_cluster
    }.items():
        # Cloud versions
        metrics[f"{name}_a1"] = func(cloud_a1)
        metrics[f"{name}_a2"] = func(cloud_a2)

        # Matrix full
        metrics[f"{name}_A1"] = func(mat_A1)
        metrics[f"{name}_A2"] = func(mat_A2)

    return metrics
