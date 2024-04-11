import pandas as pd
import matplotlib.pyplot as plt
import os

# Definir la carpeta de la que leeremos los archivos CSV
FOLDER_PATH = './data/benchmark/virtual/interpolated'

# Definir las columnas para las cuales queremos generar las gráficas
COLUMNS_TO_PLOT = ['cpu_usage_latest', 'memory_usage_gb_latest']

# Leer todos los archivos CSV de la carpeta especificada
files = os.listdir(FOLDER_PATH)
csv_files = [file for file in files if file.endswith('.csv')]

# Leer cada archivo CSV y almacenar sus DataFrames en una lista
data_frames = [pd.read_csv(os.path.join(FOLDER_PATH, csv_file)) for csv_file in csv_files]

# Por cada columna de interés, generar una gráfica
for column in COLUMNS_TO_PLOT:
    plt.figure(figsize=(14, 8))  # Inicializar una nueva figura
    
    # Verificar primero si la columna existe en cada DataFrame
    for df in data_frames:
        if column in df.columns:
            plt.plot(df.index, df[column], label=f'{column} in {df.columns[0]}', alpha=0.7)
    
    plt.xlabel('Time (arbitrary units)')
    plt.ylabel(f'{column} (units)')
    plt.title(f'{column} Over Time')
    plt.legend()
    plt.show()
