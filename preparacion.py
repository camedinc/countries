# Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

# Funciones y clases
from core.estadistica import Correlacion

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

# Types
print(df.dtypes)

df=df.astype({  'country': object,
                'child_mort': float,
                'exports': float,
                'health': float,
                'imports': float,
                'income': float,
                'inflation': float,
                'life_expec': float,
                'total_fer': float,
                'gdpp': float})

# Calidad
print("\nNull:")
print(df.isna().sum())

print("\nDuplicados:")
print(df.duplicated().sum())

print("\nNuméricas:")
print(df.describe().T)

print("\nCategóricas:")
# print(df.describe(include=['object']).T)

# Features
# df=df.drop(['Unnamed: 0'], axis=1)

print("\nTipos finales:")
print(df.dtypes)

print("\nData:")
print(df)

# Balance (sólo supervisados)
# print("\nBalance de clases:")
# print(balance(df['quality']))

# Reagrupar (sólo supervisados)
# df['quality_class'] = df['quality'].apply(
#    lambda x: 'Medium' if x in [3, 4, 5] else 'High'
#)

# print(balance(df['quality_class']))

# OHE (sólo categóricas)
# df=ohe(df)
# print(df.columns)

# Estandarización (numéricas)
# df_num=df.drop(['country'], axis=1)
# df_cat=df['country']

df_numericas=df.select_dtypes(include='number')
df_categoricas=df.select_dtypes(exclude='number')

# Estandarizar las columnas numéricas
scaler = StandardScaler()
numericas_estandarizadas = pd.DataFrame(
    scaler.fit_transform(df_numericas),
    columns=df_numericas.columns
)

# Combina con las categóricas para reconstruir el df
df_final = pd.concat([numericas_estandarizadas, df_categoricas.reset_index(drop=True)], axis=1)

# Mostrar resultado
print("Datos finales limpios")
print(df_final)

# Correlación
print("\nMatriz de correlación:")
correlacion=Correlacion(df_numericas)
matriz=correlacion.matriz_correlacion()
print(matriz)

print("\nGráfica de correlación:")
fig=correlacion.grafica_correlacion()
fig.savefig(os.path.join(path_imagenes,'1_correlacion.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# Guarda la base depurada y estandarizada
# Escritura
path_data_clean=os.path.join(path_datos,'countries-clean.csv')
df_final.to_csv(path_data_clean, sep=',', index=False, encoding='utf-8')