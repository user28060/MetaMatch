import numpy as np

def vector_to_point_cloud(v, window_size=5):
    return np.array([v[i:i+window_size] for i in range(len(v) - window_size)])
