# Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Funciones y clases


# Directorio de imágenes
root='C:\\Users\\camed\\OneDrive\\Documentos\\Git\\countries'
carpeta_imagenes='imagenes'
path_imagenes=os.path.join(root, carpeta_imagenes)
os.makedirs(path_imagenes, exist_ok=True)

# Directorio datos
root='C:\\Users\\camed\\OneDrive\\Documentos\\Git\\countries'
carpeta_datos='datos'
path_datos=os.path.join(root, carpeta_datos)
path_datos_countries=os.path.join(path_datos,'Country-data.csv')
print(path_datos_countries)

df = pd.read_csv(path_datos_countries)
print(df.head(3))
print(df.shape)

'''
# Types
print(df.dtypes)
df=df.astype({  'fixed acidity': float,
                'volatile acidity': float,
                'citric acid': float,
                'residual sugar': float,
                'chlorides': float,
                'free sulfur dioxide': float,
                'total sulfur dioxide': float,
                'density': float,
                'pH': float,
                'sulphates': float,
                'alcohol': float,
                'quality': int})


# Calidad
print("\nNull:")
print(df.isna().sum())

print("\nDuplicados:")
print(df.duplicated().sum())


print("\nNuméricas:")
print(df.describe().T)

print("\nCategóricas:")
#print(df.describe(include=['object']).T)

# Features
df=df.drop(['Unnamed: 0'], axis=1)

print("\nTipos finales:")
print(df.dtypes)

print("\nData:")
print(df)

# Balance
print("\nBalance de clases:")
print(balance(df['quality']))

# Reagrupar
df['quality_class'] = df['quality'].apply(
    lambda x: 'Medium' if x in [3, 4, 5] else 'High'
)

print(balance(df['quality_class']))

# OHE
df=ohe(df)
print(df.columns)

# Correlación
print("\nMatriz de correlación:")
correlacion=Correlacion(df)
matriz=correlacion.matriz_correlacion()
print(matriz)

print("\nGráfica de correlación:")
fig=correlacion.grafica_correlacion()
fig.savefig(os.path.join(path_imagenes,'1_correlacion.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# Guarda la base depurada
# Escritura
path_data_red_clean=os.path.join(path_datos,'winequality-red-clean.csv')
df.to_csv(path_data_red_clean, sep=',', index=True, encoding='utf-8')
'''