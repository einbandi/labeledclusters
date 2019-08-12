import numpy as np
from scipy.spatial.distance import cdist, pdist

class Cluster:

    """Container class for single clusters"""

    def __init__(self, data):
        self.data = data

    def numpy(self):
        return self.data

    def centroid(self):
        return np.mean(self.data, axis=0)

    def mean_pdist(self, metric='euclidean'):
        return np.mean(pdist(self.data, metric=metric))

    def max_diam(self, metric='euclidean'):
        return np.max(pdist(self.data, metric=metric))

    def dispersion(self, q=2):
        dists = np.linalg.norm(self.data - self.centroid())
        return np.power(np.sum(np.power(dists, q)) / len(self.data), 1/q)

    def mean_dist(self, point, metric='euclidean'):
        return np.mean(cdist([point], self.data, metric=metric))