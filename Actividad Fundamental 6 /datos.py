# Dataset escogido sobre tipos de pingüinos.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Leer el CSV de los datos y eliminar los valores nulos.
df = pd.read_csv('AF6\\pinguinos.csv')
df = df.dropna()

# Separando columnas categóricas y numéricas. 
col_cat = ['especie', 'isla', 'sexo']
col_num = ['long_culmen', 'prof_culmen', 'long_aleta', 'masa_corp']
X_cat = df[col_cat]

# Crear el encoder y convertir a DataFrame.
encoder = OneHotEncoder(drop = 'first', sparse_output = False)
X_encoded = encoder.fit_transform(X_cat)
cat_columns = encoder.get_feature_names_out(col_cat)
df_cat_encoded = pd.DataFrame(X_encoded, columns = cat_columns, index = df.index)

# Seleccionar las columnas de variable independiente (X) y dependiente (y)
X = df[['long_aleta']]
y = df[['masa_corp']]

# Normalizar los datos y hacerlos dataset.
scaler = StandardScaler()
X_num_scaled_array = scaler.fit_transform(df[col_num]) 
df_num_scaled = pd.DataFrame(X_num_scaled_array, columns = col_num, index = df.index)

# Concatenar
df_final = pd.concat([df_num_scaled, df_cat_encoded, y], axis = 1) 

# Separar datos en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Crear el modelo y entrenarlo.
modelo = LinearRegression()
modelo.fit(X_train, y_train) 

# Hacer predicciones y evaluar el modelo.
y_pred = modelo.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Interpretación
if r2 == 1: interpretacion = 'Modelo perfecto'
elif r2 > 0.7: interpretacion = 'Buena precisión'
elif r2 >= 0.3: interpretacion = 'Modelo regular'
else: interpretacion = 'Modelo poco confiable'

# Mostrar resultados
print(f'\nModelo aprendido: y = {modelo.coef_[0][0]:.2f}x + {modelo.intercept_[0]:.2f}')
print(f'R^2: {r2:.4f} ({interpretacion})')
print(f'Error cuadrático medio (MSE): {mse:.2f}')
print(f'Raíz del error cuadrático medio (RMSE): {rmse:.2f}\n')

# Usa df_final que ya creaste arriba
print(df_final)

# Gráfico (con datos de prueba)
plt.scatter(X_test, y_test, color = 'red', label = 'Datos reales')
plt.plot(X_test, y_pred, color = 'blue', label = 'Predicción')
plt.xlabel('Longitud de aleta')
plt.ylabel('Masa corporal')
plt.title('Regresión lineal: masa_corp vs long_aleta')
plt.legend()
plt.show()
