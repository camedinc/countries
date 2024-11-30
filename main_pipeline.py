# Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import optuna
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
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
model_dict={'K-Means': {
                        'model':KMeans,
                        'param_grid':{
                        'n_clusters':[2, 3, 4, 5],
                         'init': ['k-means++', 'random']
                        }
                    },
            'Hierarchical (Aglomerativo)': {
                        'model':AgglomerativeClustering,
                        'param_grid':{
                        'n_clusters':[2, 3, 4, 5],
                        'linkage': ['ward', 'complete', 'average']
                        }
                    }
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


