import pandas as pd

df = pd.read_csv('AF4\\videogames.csv')

# Función para mostrar los datos originales llamada desde 'datos.py'
def mostrar_original():
    print(df)
    print("\n")