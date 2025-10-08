import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

df = pd.read_csv('AF4\\videogames.csv')

# Reemplazar 'tbd' en user_score con NaN y convertir user_score a float
df['user_score'] = df['user_score'].replace('tbd', np.nan)
df['user_score'] = pd.to_numeric(df['user_score'], errors = 'coerce')

col_num = ['year', 'critic_score', 'critic_count', 'user_score'] # Columnas numéricas a normalizar
datos_num = df[col_num].copy()

# Manejar valores faltantes (imputar con la mediana para mantener robustez)
for col in col_num:
    datos_num[col] = datos_num[col].fillna(datos_num[col].median())

# Aplicar Min-Max Scaling
scaler = MinMaxScaler()
datos_normalizados = scaler.fit_transform(datos_num)

# Convertir los datos normalizados a un DataFrame
df_normalizado = pd.DataFrame(datos_normalizados, columns = col_num)

# Combinar con las columnas categóricas
columnas_categoricas = ['id', 'name', 'platform', 'genre', 'publisher', 'developer', 'rating']
df_final = pd.concat([df[columnas_categoricas].reset_index(drop = True), df_normalizado], axis = 1)

# Guardar el resultado en un nuevo CSV
df_final.to_csv('videogames_normalizado.csv', index = False)

# Función para mostrar los datos normalizados llamada desde 'datos.py'
def mostrar_normalizado():
    print(df_final)
    print("\n")