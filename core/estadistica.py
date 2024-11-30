# Librerías
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Clase correlacion
class Correlacion:
    def __init__(self, df):
        self.df=df
        self.matriz=None
        self.heat_map=None
    def matriz_correlacion(self):
        print("Accede a matriz")
        self.matriz=self.df.corr()
        return self.matriz
    def grafica_correlacion(self):
        if self.matriz is None:
            print("Ejecuta primero matriz_correlación")
        else:
            fig, ax=plt.subplots()
            self.heat_map=sns.heatmap(self.matriz, annot=True, cmap='viridis')
        return fig