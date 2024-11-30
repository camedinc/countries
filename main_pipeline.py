# Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import optuna
from sklearn.pipeline import Pipeline
from sklearn.cluster import (KMeans, AgglomerativeClustering, DBSCAN, OPTICS, SpectralClustering, Birch, AffinityPropagation, MeanShift)
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
from hdbscan import HDBSCAN
from kmodes.kprototypes import KPrototypes
from minisom import MiniSom
from sklearn.metrics import silhouette_score

# Funciones y clases

# Data
# Directorio datos
root='C:\\Users\\camed\\OneDrive\\Documentos\\Git\\countries'
carpeta_datos='datos'
path_datos=os.path.join(root, carpeta_datos)
path_datos_countries=os.path.join(path_datos,'countries-clean.csv')

df = pd.read_csv(path_datos_countries)
print(df.head(3))
print(df.shape)

# Implementación de Pipeline (requiere modelos estándar, no personalizados)
#--------------------------------------------------------------------------------------

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
#        "model": MiniSom,  # Requiere paquetes específicos como Minisom o SOMPY.
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
#        "model": None,  # Requiere paquetes como PyTorch/TensorFlow para autoencoders.
#        "param_grid": {
#            "hidden_layers": [(128, 64), (256, 128, 64)],
#            "learning_rate": [0.001, 0.01],
#        },
#    },
}

# Datos
data = df.drop(['country'], axis=1)

# Lista que almacena resultados
resultados=[]

# Ciclo que optimiza cada modelo
for model_name, config in model_dict.items(): # Recorre el diccionario
    def objetive(trial):
        model_class=config['model']               # Almacena las componentes en variables  
        param_grid=config['param_grid']
        # Crea los parámetros a partir del grid
        params={key: trial.suggest_categorical(key, values) for key, values in param_grid.items()}
        model=model_class(**params)

        # Ajustar el modelo
        if hasattr(model, "fit_predict"):
            labels=model.fit_predict(data)
        else:
            labels=model.fit(data).predict(data)

          # Validar número de etiquetas
        n_labels = len(set(labels))
        if n_labels < 2 or n_labels >= len(data):  # Verifica el rango válido
            return -1  # Penaliza configuraciones inválidas
        
        # Calcula el score de sliueta
        if len(set(labels)) > 1:                   # Evita errores de cluster único
            return silhouette_score(data, labels)
        return -1                                  # Penaliza configuraciones inválidad

    # Crea el estudio y lo optimiza
    study = optuna.create_study(direction="maximize")
    study.optimize(objetive, n_trials=30)          # n_trials según recursos

    # Guardar resultados
    resultados.append({"model":model_name,
                       "best_params":study.best_params,
                       "best_score":study.best_value})
# Imprime los resultados
for result in resultados:
    print(f"Modelo: {result['model']}")
    print(f"Mejores parámetros: {result['best_params']}")
    print(f"Mejor puntuación: {result['best_score']:.4f}")