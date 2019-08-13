import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from itertools import combinations
from clusters import Cluster


def centroid_dist(a, b, metric='euclidean'):
    return cdist([a.centroid()], [b.centroid()], metric=metric).item()


def mean_dist(a, b, metric='euclidean'):
    return np.mean(cdist(a.numpy(), b.numpy(), metric='euclidean'))


def davies_bouldin_sim(a, b, metric='euclidean', q=2):
    return (a.dispersion(q=q) + b.dispersion(q=q)) / centroid_dist(a, b, metric=metric)


def estimate_density(x, y, x0, y0, sigma):

    xp = np.power(np.atleast_3d(x0) - x, 2)
    yp = np.power(np.atleast_3d(y0) - y, 2)
    exp = np.exp(-1 / (2*sigma) * (xp + yp))
    result = 1 / (2*np.pi*sigma) * np.sum(exp, axis=2) / len(x)

    maxdim = max(len(np.shape(x0)), len(np.shape(y0)))

    funcs = {
        0: lambda z: z.item(),
        1: lambda z: z[0],
        2: lambda z: z,
    }

    return funcs[maxdim](result)


def raw__data_to_table(data, labels):

    num_pts, num_samples, num_dim = data.shape

    ids = np.arange(num_pts).repeat(num_samples)
    labels = labels.repeat(num_samples)
    coords = data.reshape(-1, num_dim).transpose()
    samples = np.tile(np.arange(num_samples), num_pts)

    data_dict = {
        'id': ids,
        'label': labels,
        'sample': samples
    }

    for dim, xs in enumerate(coords):
        data_dict['x' + str(dim)] = xs

    return pd.DataFrame(data=data_dict)

def get_values(df):
    return df.drop(['id', 'label', 'sample'], axis=1).values


class LabeledSampledOutputs:

    """Container class for sampled neural network outputs with known class labels."""

    def __init__(self, data, labels):

        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        if not isinstance(labels, np.ndarray):
            labels = np.asarray(labels)

        self.data = data

        if len(labels) == len(data):
            self.labels = labels
        else:
            print('{}::clbls: Length of label list must match length of data.'.format(
                self.__class__.__name__))

        self.num_classes = len(set(labels))
        self.num_samples = data.shape[1]
        self.num_dims = data.shape[2]

        self.df = raw__data_to_table(data, labels)

    def __len__(self):
        return len(self.data)

    def instances_per_class(self):
        labels, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(labels, counts))

    def get_class_clusters(self):
        return [Cluster(get_values(self.df[self.df['label'] == label])) for label in set(self.labels)]

    def get_class_cluster(self, label):
        return Cluster(get_values(self.df[self.df['label'] == label]))

    def get_instance_cluster(self, index):
        # alternatively simply Cluster(data[id])
        return Cluster(get_values(self.df[self.df['id'] == index]))

    def get_class_cluster_without_instance(self, index):
        return Cluster(get_values(self.df[(self.df['label'] == self.labels[index]) & (self.df['id'] != index)]))

    def inter_class_mean_dist(self, class_a, class_b, metric='euclidean'):
        a = self.get_class_cluster(class_a)
        b = self.get_class_cluster(class_b)
        return mean_dist(a, b, metric=metric)

    def inter_class_centroid_dist(self, class_a, class_b, metric='euclidean'):
        a = self.get_class_cluster(class_a)
        b = self.get_class_cluster(class_b)
        return centroid_dist(a, b, metric=metric)

    def intra_class_mean_dist(self, label, metric='euclidean'):
        return self.get_class_cluster(label).mean_pdist(metric=metric)

    def class_diam(self, label, metric='euclidean'):
        return self.get_class_cluster(label).max_diam(metric=metric)

    def instance_class_mean_dist(self, index, label, metric='euclidean'):
        i = self.get_instance_cluster(index)
        if self.labels[index] == label:
            c = self.get_class_cluster_without_instance(index)
        else:
            c = self.get_class_cluster(label)
        return mean_dist(i, c, metric=metric)

    def instance_class_mean_dists(self, index, metric='euclidean'):
        dists = dict()
        for c in set(self.labels):
            dists[c] = self.instance_class_mean_dist(index, c, metric=metric)
        return dists

    def closest_class(self, index, metric='euclidean'):
        dists = self.instance_class_mean_dists(index, metric=metric)
        dists = list(dists.values())
        index = np.argmin(dists)
        return list(set(self.labels))[index]

    def closest_class_dist(self, index, metric='euclidean'):
        dists = self.instance_class_mean_dists(index, metric=metric)
        dists = list(dists.values())
        return np.min(dists)

    def closest_other_class_dist(self, index, metric='euclidean'):
        dists = self.instance_class_mean_dists(index, metric=metric)
        dists = list(dists.values())
        label_index = list(set(self.labels)).index(self.labels[index])
        dists[label_index] = np.inf
        return np.min(dists)

    def silhouette_index(self, index, metric='euclidean'):
        a = self.instance_class_mean_dist(index, self.labels[index], metric=metric)
        b = self.closest_other_class_dist(index, metric=metric)
        return (b-a) / np.max([a, b])

    def silhouette_indices(self, metric='euclidean'):
        return np.array([self.silhouette_index(i, metric=metric) for i in range(len(self))])


    def class_overlap(self, x, y, sigma):
        if(self.num_dims != 2):
            print('Overlap currently only implemented for 2 dimensional clusters!')
            return float('nan')

        densities = [estimate_density(
            c.data[:, 0], c.data[:, 1], x, y, sigma) for c in self.get_class_clusters()]
        return np.product(np.asarray(list(combinations(densities, 2))), axis=1).sum(axis=0)

    def mean_overlap(self, index, sigma):
        if(self.num_dims != 2):
            print('Overlap currently only implemented for 2 dimensional clusters!')
            return float('nan')

        c = self.get_instance_cluster(index)
        return np.mean(self.class_overlap(c.data[:, 0], c.data[:, 1], sigma))

    def overlap_map(self, x_range, y_range, sigma):
        if(self.num_dims != 2):
            print('Overlap map currently only implemented for 2 dimensionsl clusters!')
            return float('nan')

        (xmin, xmax, xstep) = x_range
        (ymin, ymax, ystep) = y_range
        x_space = np.linspace(xmin, xmax, xstep)
        y_space = np.linspace(ymin, ymax, ystep)
        x, y = np.meshgrid(x_space, y_space)

        print(x)

        return self.class_overlap(x, y, sigma)
