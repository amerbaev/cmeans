from skfuzzy.cluster import cmeans
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import pandas as pd
import seaborn as sns


def process_data(data, centers):
    cntr, u, u0, d, jm, p, fpc = cmeans(data, centers, 2, error=0.005, maxiter=10000, init=None)
    cluster_membership = np.argmax(u, axis=0)

    data = pd.DataFrame(data=data)
    data['target'] = cluster_membership

    g = sns.FacetGrid(data, hue='target', palette='tab20', size=5)
    g.map(plt.scatter, 0, 1, s=100, linewidth=.5, edgecolor='white')
    g.add_legend()
    # for i in range(centers):
    #     plt.scatter(data[cluster_membership == i])
    # for pt in cntr:
    #     plt.plot(pt[0], pt[1], 'rs')
    plt.show()


def main():
    n_samples = 10000
    random_state = 170
    transformation = [[0.6, -0.6], [-0.4, 0.8]]

    models = [
        {
            'name': 'Far Blobs',
            'X':
                datasets.make_blobs(n_samples=n_samples, centers=25, random_state=0, center_box=(-10000, 10000),
                                    cluster_std=50)[0],
            'k': 25
        },
        {
            'name': 'Noisy Circles',
            'X':
                datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)[0],
            'k': 2
        },
        {
            'name': 'Noisy Moons',
            'X': datasets.make_moons(n_samples=n_samples, noise=.05)[0],
            'k': 2
        },
        {
            'name': 'Blobs',
            'X': datasets.make_blobs(n_samples=n_samples, random_state=8)[0],
            'k': 3
        },
        {
            'name': 'No structure',
            'X': np.random.rand(n_samples, 2),
            'k': 2
        },
        {
            'name': 'Anisotropicly distributed data',
            'X': np.dot(datasets.make_blobs(n_samples=n_samples, random_state=random_state)[0], transformation),
            'k': 2
        },
        {
            'name': 'Blobs with varied variances',
            'X': datasets.make_blobs(n_samples=n_samples,
                                     cluster_std=[1.0, 2.5, 0.5],
                                     random_state=random_state)[0],
            'k': 2
        },
        {
            'name': '2 features, 1 informative, 1 cluster',
            'X': datasets.make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=1,
                                              n_clusters_per_class=1)[0],
            'k': 2
        },
        {
            'name': '2 features, 2 informative, 1 cluster',
            'X': datasets.make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=2,
                                              n_clusters_per_class=1)[0],
            'k': 3
        },
        {
            'name': '2 features, 2 informative',
            'X': datasets.make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=2)[0],
            'k': 2
        },
        {
            'name': '2 features, 2 informative, 2 cluster, 3 classes',
            'X': datasets.make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=2,
                                              n_clusters_per_class=1, n_classes=3)[0],
            'k': 3
        },
        {
            'name': '2 features, 5 centers',
            'X': datasets.make_blobs(n_samples=500, n_features=2, centers=5)[0],
            'k': 4,
            'dist_metric': 'manhattan'
        },
        {
            'name': '2 features, 6 classes',
            'X': datasets.make_gaussian_quantiles(n_samples=500, n_features=2, n_classes=6)[0],
            'k': 6
        },
        {
            'name': 'Circles',
            'X': datasets.make_circles(n_samples=500, factor=0.5)[0],
            'k': 4,
            'dist_metric': 'manhattan'
        },
    ]

    for model in models:
        process_data(model['X'], model['k'])

if __name__ == '__main__':
    main()