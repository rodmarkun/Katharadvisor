import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import os

# Carpeta que contiene los archivos CSV
FOLDER_PATH = './data/'
COLUMNS_TO_INTERPOLATE = ['cpu_usage_latest', 'memory_usage_gb_lastest']

# Función modificada para interpolar múltiples columnas
def interpolate_dataframe(df, target_length, columns_to_interpolate):
    interpolated_columns = {}
    x_new = np.linspace(0, len(df) - 1, target_length)
    
    for column_name in columns_to_interpolate:
        x_original = np.arange(len(df))
        y_original = df[column_name].values
        interpolation_function = interp1d(x_original, y_original, kind='linear')
        y_interpolated = interpolation_function(x_new)
        interpolated_columns[column_name] = y_interpolated
        
    return pd.DataFrame(interpolated_columns)

# Encuentra la longitud máxima entre todos los archivos CSV
max_length = 0
for file in os.listdir(FOLDER_PATH):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(FOLDER_PATH, file))
        if len(df) > max_length:
            max_length = len(df)

# Procesa cada archivo CSV en la carpeta
for file in os.listdir(FOLDER_PATH):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(FOLDER_PATH, file))
        interpolated_df = interpolate_dataframe(df, max_length, COLUMNS_TO_INTERPOLATE)
        # Define un nuevo nombre de archivo para el CSV interpolado
        new_file_name = os.path.splitext(file)[0] + '_interpolated.csv'
        # Guarda el DataFrame interpolado
        interpolated_df.to_csv(os.path.join(FOLDER_PATH, new_file_name), index=False)
