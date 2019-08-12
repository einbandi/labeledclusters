import numpy as np
from scipy.spatial.distance import cdist, pdist
from itertools import combinations
import clusters


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
    
    return  funcs[maxdim](result)


class LabeledSampledOutputs:

    """Container class for sampled neural network outputs with known class labels."""

    def __init__(self, data, class_labels=None):
        self.data = data
        if class_labels is None:
            self.class_labels = [i for i in range(len(data))]
        else:
            if len(class_labels) == len(data):
                self.class_labels = class_labels
            else:
                print('{}::clbls: Length of label list must match length of data.'.format(
                    self.__class__.__name__))

        samples = set()
        dims = set()
        for label in self.data:
            for instance in label:
                samples.add(len(instance))
                for sample in instance:
                    dims.add(len(sample))

        if len(samples) > 1:
            print('{}::sampl: Number of samples must be the same for each instance.'.format(
                self.__class__.__name__))

        if len(dims) > 1:
            print('{}::dims: Number of dimensions for each sample must be the same.'.format(
                self.__class__.__name__))

        self.num_classes = len(data)
        self.num_samples = next(iter(samples))
        self.num_dims = next(iter(dims))

    def instances_per_class(self):
        ipc = dict()
        for label, instances in enumerate(self.data):
            class_label = self.class_labels[label]
            ipc[class_label] = len(instances)
        return ipc

    def convert_index(self, index):
        ipc = self.instances_per_class()
        if type(index) is tuple and len(index) == 2:
            class_label, instance = index
            result = 0
            for i in ipc:
                if(i != class_label):
                    result += ipc[i]
                else:
                    break
            return result + instance
        else:
            result = index
            for i in ipc:
                if(result - ipc[i] < 0):
                    class_label = i
                    break
                else:
                    result -= ipc[i]
            return (class_label, result)
        
    def get_class_clusters(self):
        return [Cluster(c.reshape(-1, self.num_dims)) for c in self.data]

    def get_class_cluster(self, class_label):
        index = self.class_labels.index(class_label)
        return Cluster(self.data[index].reshape(-1, self.num_dims))

    def get_instance_cluster(self, instance_spec):
        class_label, instance_index = instance_spec
        class_index = self.class_labels.index(class_label)
        return Cluster(self.data[class_index, instance_index])

    def get_class_cluster_without_instance(self, instance_spec):
        class_label, instance_index = instance_spec
        class_index = self.class_labels.index(class_label)
        class_cluster = self.data[class_index]
        removed = np.delete(class_cluster, instance_index, axis=0)
        return Cluster(removed.reshape(-1, self.num_dims))

    def inter_class_mean_dist(self, class_a, class_b, metric='euclidean'):
        a = self.get_class_cluster(class_a).numpy()
        b = self.get_class_cluster(class_b).numpy()
        return mean_dist(a, b, metric=metric)

    def inter_class_centroid_dist(self, class_a, class_b, metric='euclidean'):
        a = self.get_class_cluster(class_a)
        b = self.get_class_cluster(class_b)
        return centroid_dist(a, b, metric=metric)

    def intra_class_mean_dist(self, class_label, metric='euclidean'):
        return self.get_class_cluster(class_label).mean_pdist(metric=metric)

    def class_diam(self, class_label, metric='euclidean'):
        return self.get_class_cluster(class_label).max_diam(metric=metric)

    def instance_class_mean_dist(self, instance_spec, class_label, metric='euclidean'):
        i = self.get_instance_cluster(instance_spec)
        if instance_spec[1] == class_label:
            c = self.get_class_cluster_without_instance(instance_spec)
        else:
            c = self.get_class_cluster(class_label)
        return mean_dist(i, c, metric=metric)

    def instance_class_mean_dists(self, instance_spec, metric='euclidean'):
        dists = []
        for c in self.class_labels:
            dists.append(self.instance_class_mean_dist(
                instance_spec, c, metric=metric))
        return np.asarray(dists)

    def closest_class(self, instance_spec, metric='euclidean'):
        index = np.argmin(self.instance_class_mean_dists(
            instance_spec, metric=metric))
        return self.class_labels[index]

    def closest_class_dist(self, instance_spec, metric='euclidean'):
        return np.min(self.instance_class_mean_dists(instance_spec, metric=metric))

    def closest_other_class_dist(self, instance_spec, metric='euclidean'):
        dists = self.instance_class_mean_dists(
            instance_spec, metric='euclidean')
        dists[self.class_labels.index(instance_spec[0])] = np.inf
        return np.min(dists)

    def silhouette_index(self, instance_spec, metric='euclidean'):
        a = self.instance_class_mean_dist(
            instance_spec, instance_spec[0], metric=metric)
        b = self.closest_other_class_dist(instance_spec, metric=metric)
        return (b-a) / np.max([a, b])

    def silhouette_indices(self, metric='euclidean'):
        ipc_dict = self.instances_per_class()
        result = []
        for label in ipc_dict:
            class_result = []
            for i in range(ipc_dict[label]):
                class_result.append(self.silhouette_index(
                    (label, i), metric=metric))
            result.append(class_result)
        return result
    
    def class_overlap(self, x, y, sigma):
        if(self.num_dims != 2):
            print('Overlap currently only implemented for 2 dimensional clusters!')
            return float('nan')
        
        densities = [estimate_density(c.data[:,0], c.data[:,1], x, y, sigma) for c in self.get_class_clusters()]
        return np.product(np.asarray(list(combinations(densities, 2))), axis=1).sum(axis=0)
    
    def overlap_map(self, x_range, y_range, sigma):
        if(self.num_dims != 2):
            print('Overlap map currently only implemented for 2 dimensionsl clusters!')
            return float('nan')
        
        (xmin, xmax, xstep) = x_range
        (ymin, ymax, ystep) = y_range
        x_space = np.linspace(xmin, xmax, xstep)
        y_space = np.linspace(ymin, ymax, ystep)
        x,y = np.meshgrid(x_space, y_space)
        
        return self.class_overlap(x, y, sigma)
