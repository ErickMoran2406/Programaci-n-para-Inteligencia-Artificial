import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Cargar el archivo 'penguins.csv' en un DataFrame y elimina valores nulos.
df = pd.read_csv('PIA\\penguins.csv')
df = df.dropna() 

# Eliminar la columna 'id', ya que no es útil en este caso.
df = df.drop(columns = ['id', 'year'])

# Definir las variables predictoras (X) y la variable objetivo (y)
X = df.drop(columns = ['body_mass_g'])
y = df['body_mass_g']

# DataFrame antes del proceso.
print(f'DataFrame antes del proceso:\n{df}\n')

# Definir columnas categóricas y numéricas.
col_cat = ['species', 'island', 'sex']
col_num = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']

# Crear el preprocesador, aplica diferentes transformaciones según la columna.
preprocessor = ColumnTransformer( transformers = [
    # Estandariza las variables numéricas,
    ('num', StandardScaler(), col_num), 
    # Codifica variables categóricas con One-Hot Encoding,
    ('cat', OneHotEncoder(drop = 'first'), col_cat) 
], remainder = 'passthrough')

# Limpia, transforma los datos y entrena el modelo de regresión lineal.
pipeline = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Entrenamiento 70% - Prueba 30%.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Entrenar, predecir, realizar y mostrar métricas.
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_temp)

r2 = r2_score(y_temp, y_pred)
mae = mean_absolute_error(y_temp, y_pred)
mse = mean_squared_error(y_temp, y_pred)
rmse = np.sqrt(mse)

print(f"R^2: {r2:.4f}")
print(f"MAE: {mae:.2f} g")
print(f'MSE: {mse:.2f} g^2')
print(f'RMSE: {rmse:.2f} g')

# Mostrar las filas de 'X_train' después del preprocesamiento.
X_transformado = pd.DataFrame(
    pipeline.named_steps['preprocessor'].transform(X_train),
    columns = col_num + list(pipeline.named_steps['preprocessor'].
                             named_transformers_['cat'].get_feature_names_out(col_cat))
); 

# Mostrar el DataFrame procesado.
print(f'\nDataFrame procesado:\n{X_transformado}\n')

# Aplicar el preprocesamiento a todo el conjunto de datos X,
X_transf = pipeline.named_steps['preprocessor'].transform(X)

# Nombres de las columnas transformadas
cols_transf = (
    col_num + list(pipeline.named_steps['preprocessor']
         .named_transformers_['cat']
         .get_feature_names_out(col_cat))
)

# Convertir datos transformados a DataFrame.
X_transf_df = pd.DataFrame(X_transf, columns = cols_transf, index = X.index)

# Concatenar el DataFrame original con las columnas transformadas.
df_completo = pd.concat([df, X_transf_df], axis = 1)

# Mostrar el DataFrame completo.
print(f"\nDataFrame completo:\n{df_completo}\n")

# Crear un gráfico de dispersión para comparar valores reales y predichos.
plt.scatter(y_temp, y_pred, color = 'red')
plt.plot([y_temp.min(), y_temp.max()],  [y_temp.min(), y_temp.max()], color = 'blue')
plt.xlabel('Peso real (g)')
plt.ylabel('Peso predicho (g)')
plt.title(f'Regresión Lineal - Predicción del peso corporal de pingüinos\n'
          f'R^2 = {r2:.4f} | MAE = {mae:.1f} g | RMSE = {rmse:.1f} g')
plt.grid()
plt.show()
