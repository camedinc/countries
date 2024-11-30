# Librerías
from sklearn.cluster import (KMeans, AgglomerativeClustering, DBSCAN, OPTICS, SpectralClustering, Birch, AffinityPropagation, MeanShift)
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
from hdbscan import HDBSCAN
from kmodes.kprototypes import KPrototypes
from minisom import MiniSom

# Diccionario de modelos y parámetros 
model_dict = {
    "K-Means": {
        "model": KMeans,
        "param_grid": {
            "n_clusters": [2, 3, 4, 5, 6],
            "init": ["k-means++", "random"],
            "max_iter": [300, 500, 1000],
        },
    },
    "K-Median": {
        "model": KMeans,  # Aproximación porque scikit-learn no incluye K-Median; usar KMeans con otra métrica.
        "param_grid": {
            "n_clusters": [2, 3, 4, 5, 6],
            "init": ["k-means++", "random"],
            "algorithm": ["elkan"],
        },
    },
    "K-Medoids": {
        "model": KMedoids,
        "param_grid": {
            "n_clusters": [2, 3, 4, 5, 6],
            "metric": ["euclidean", "manhattan"],
        },
    },
#    "K-Prototipos": {                       # Requiere columnas categóricas
#        "model": KPrototypes,
#        "param_grid": {
#            "n_clusters": [2, 3, 4, 5, 6],
#            "init": ["Huang", "Cao"],
#            "n_init": [10, 20],
#        },
#    },
    "Hierarchical (Aglomerativo)": {
        "model": AgglomerativeClustering,
        "param_grid": {
            "n_clusters": [2, 3, 4, 5],
            "linkage": ["ward", "complete", "average", "single"],
        },
    },
    "DBSCAN": {
        "model": DBSCAN,
        "param_grid": {
            "eps": [0.1, 0.5, 1.0, 1.5],
            "min_samples": [3, 5, 10],
        },
    },
    "HDBSCAN": {
        "model": HDBSCAN,
        "param_grid": {
            "min_cluster_size": [2, 5, 10],
            "min_samples": [1, 3, 5],
            "cluster_selection_epsilon": [0.1, 0.5, 1.0],
        },
    },
    "OPTICS": {
        "model": OPTICS,  # OPTICS puede ser implementado con scikit-learn o `sklearn.cluster.OPTICS`.
        "param_grid": {
            "min_samples": [3, 5, 10],
            "eps": [0.5, 1.0, 2.0],
            "xi": [0.05, 0.1, 0.2]
        },
    },
    "Mean-Shift": {
        "model": MeanShift,
        "param_grid": {
            "bandwidth": [0.5, 1.0, 2.0, 3.0],
        },
    },
    "Gaussian Mixture (GMM)": {
        "model": GaussianMixture,
        "param_grid": {
            "n_components": [2, 3, 4],
            "covariance_type": ["full", "tied", "diag", "spherical"],
        },
    },
    "Spectral Clustering": {
        "model": SpectralClustering,
        "param_grid": {
            "n_clusters": [2, 3, 4, 5],
            "affinity": ["rbf", "nearest_neighbors"],
        },
    },
    "BIRCH": {
        "model": Birch,
        "param_grid": {
            "n_clusters": [2, 3, 4, 5],
            "threshold": [0.3, 0.5, 0.7],
        },
    },
    "Affinity Propagation": {
        "model": AffinityPropagation,
        "param_grid": {
            "damping": [0.5, 0.7, 0.9],
            "preference": [-50, -25, 0, 25],
        },
    },
#    "Self-Organizing Maps (SOM)": {
#        "model": MiniSom,  # Requiere paquetes específicos como Minisom o SOMPY, su ajuste fit es especial
#        "param_grid": {
#            "x": [10, 15, 20],  # Tamaño en filas del mapa
#            "y": [10, 15, 20],  # Tamaño en columnas del mapa
#            "learning_rate": [0.01, 0.1], # Tasa de aprendizaje
#            "input_len": [5, 10, 15], # Dimensiones de los datos
#            "sigma": [0.5, 1.0], # Vecindad inicial
#        },
#        "fit_method":"train_random",
#    },
#    "Deep Clustering": {
#        "model": None,  # Requiere paquetes como PyTorch/TensorFlow para autoencoders. Debe definir la red
#        "param_grid": {
#            "hidden_layers": [(128, 64), (256, 128, 64)],
#            "learning_rate": [0.001, 0.01],
#        },
#    },
}