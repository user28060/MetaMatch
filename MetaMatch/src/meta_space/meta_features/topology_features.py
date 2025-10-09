import warnings

import numpy as np
from gudhi.bottleneck import bottleneck_distance
from gudhi.wasserstein import wasserstein_distance
from persim.persistent_entropy import persistent_entropy
from ripser import ripser

warnings.filterwarnings(
    "ignore",
    message="The input point cloud has more columns than rows; did you mean to transpose\\?",
)


def vector_to_point_cloud(v, window_size=5):
    return np.array([v[i : i + window_size] for i in range(len(v) - window_size)])


def compute_topological_summary(dgms, name):
    h0_intervals = dgms[0]
    h1_intervals = dgms[1]
    return {
        f"h0_count_{name}": len(h0_intervals),
        f"h1_count_{name}": len(h1_intervals),
        f"h0_max_lifetime_{name}": max([b[1] - b[0] for b in h0_intervals if b[1] != np.inf], default=0),
        f"h1_max_lifetime_{name}": max([b[1] - b[0] for b in h1_intervals if b[1] != np.inf], default=0),
    }


def compute_topological_metrics(v1, v2, mat_A1, mat_A2, metric_option):
    cloud_a1 = vector_to_point_cloud(v1)
    cloud_a2 = vector_to_point_cloud(v2)
    cloud_a1_a2 = np.vstack([cloud_a1, cloud_a2])

    dgms_A1 = ripser(mat_A1, metric=metric_option)["dgms"]
    dgms_A2 = ripser(mat_A2, metric=metric_option)["dgms"]
    dgms_a1 = ripser(cloud_a1, metric=metric_option)["dgms"]
    dgms_a2 = ripser(cloud_a2, metric=metric_option)["dgms"]
    dgms_a1_a2 = ripser(cloud_a1_a2, metric=metric_option)["dgms"]

    topo = {}

    for prefix, dgms in zip(
        ["A1", "A2", "a1", "a2", "a1_a2", "A1_A2"],
        [dgms_A1, dgms_A2, dgms_a1, dgms_a2, dgms_a1_a2],
    ):
        topo.update(compute_topological_summary(dgms, prefix))
        topo[f"{metric_option}_entropy_H0_{prefix}"] = persistent_entropy(dgms[0])[0]
        topo[f"{metric_option}_entropy_H1_{prefix}"] = persistent_entropy(dgms[1])[0]

    for name, (dgms1, dgms2) in {
        "a1_a2": (dgms_a1, dgms_a2),
        "A1_A2": (dgms_A1, dgms_A2),
    }.items():
        topo[f"{metric_option}_bottleneck_{name}"] = bottleneck_distance(dgms1[0], dgms2[0])
        topo[f"{metric_option}_wasserstein_{name}"] = wasserstein_distance(dgms1[0], dgms2[0])

    return topo
