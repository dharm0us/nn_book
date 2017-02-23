import mnist_loader
import random

from time import time
import numpy as np

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from collections import OrderedDict

np.random.seed(42)
sample_size = 300
n_digits = 10

def bench_k_means(estimator, name, data,labels):
    print("kaka")
    print(data.shape)
    print(labels.shape)
    t0 = time()
    estimator.fit(data)
    print(79 * '_')
    print(estimator.labels_)
    print(estimator.labels_.shape)
    print(labels.shape)
    # dict = {}
    # for x, y in np.nditer([labels, estimator.labels_]):
    #     key = str(x) + "-" + str(y)
    #     dict[key] = dict.get(key,0) + 1
    #
    #     #sort by value
    # s = [(k, dict[k]) for k in sorted(dict, key=dict.get, reverse=True)]
    # for k, v in s:
    #     print(k,v)
    #
    #     #sort by key
    # b = OrderedDict(sorted(dict.items()))
    # print(b)

    print(np.unique(estimator.labels_))
    print(np.sum(labels == estimator.labels_))
    print(np.sum(labels != estimator.labels_))
    print(79 * '_')
    print('% 9s' % 'init' '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')
    print(79 * '_')
    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_, metric='euclidean', sample_size=sample_size)))


def data_size_784():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    random.shuffle(training_data)
    training_data = training_data[:2000]

    labels = [i[1].flatten().nonzero()[0] for i in training_data]
    labels = np.asarray(labels).flatten()

    data = [i[0].flatten() for i in training_data]
    data = np.asarray(data)
    bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10), name="k-means++", data=data,labels=labels)

def data_size_64():
    digits = load_digits()
    data = scale(digits.data)
    labels = digits.target
    bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10), name="k-means++", data=data,labels=labels)

data_size_784()
data_size_64()
