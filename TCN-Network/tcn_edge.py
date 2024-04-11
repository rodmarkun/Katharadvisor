import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from kerastuner import HyperModel, RandomSearch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tcn import TCN, tcn_full_summary
import matplotlib.pyplot as plt

# Definir las rutas a las carpetas de grupo de características y etiquetas, y test
GROUP_FEATURES_PATH = 'data/benchmark/virtual/interpolated'
GROUP_LABELS_PATH = 'data/benchmark/real'
TEST_FEATURES_PATH = 'data/benchmark/test/kdata-10_interpolated.csv'
TEST_LABELS_PATH = 'data/benchmark/test/data-2024-04-10-12%3A07%3A42.csv'

# Función para cargar y preprocesar datos
def load_and_preprocess_csv(feature_csv_path, label_csv_path):
    # Carga y selecciona solo uso de CPU y memoria
    features_df = pd.read_csv(feature_csv_path)[['cpu_usage_latest', 'memory_usage_gb_latest']]
    labels_df = pd.read_csv(label_csv_path)[['cpu_usage_latest', 'memory_usage_gb_latest']]
    # Normaliza las características
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features_df)
    labels = labels_df.values
    return features_scaled, labels

# Función para procesar grupos de archivos
def process_group(feature_folder_path, label_folder_path):
    feature_files = [os.path.join(feature_folder_path, file) for file in os.listdir(feature_folder_path) if file.endswith('.csv')]
    label_files = [os.path.join(label_folder_path, file) for file in os.listdir(label_folder_path) if file.endswith('.csv')]
    
    features, labels = [], []
    # Asume la misma cantidad de archivos de características y etiquetas, en el mismo orden
    for feature_file, label_file in zip(feature_files, label_files):
        features_scaled, labels_data = load_and_preprocess_csv(feature_file, label_file)
        features.append(features_scaled)
        labels.append(labels_data)
    return np.concatenate(features), np.concatenate(labels)

# Procesar datos de entrenamiento y prueba
training_features, training_labels = process_group(GROUP_FEATURES_PATH, GROUP_LABELS_PATH)

# Asume un solo archivo de test para características y etiquetas
test_features_scaled, test_labels = load_and_preprocess_csv(TEST_FEATURES_PATH, TEST_LABELS_PATH)

# Reshape para TCN
training_features_reshaped = training_features.reshape((training_features.shape[0], training_features.shape[1], 1))
test_features_reshaped = test_features_scaled.reshape((test_features_scaled.shape[0], test_features_scaled.shape[1], 1))

# Split training and test labels for CPU and Memory
training_cpu_labels = training_labels[:, 0]
training_memory_labels = training_labels[:, 1]
test_cpu_labels = test_labels[:, 0]
test_memory_labels = test_labels[:, 1]

# Define HyperModel class (modified for single output dimension)
class TCNHyperModel(HyperModel):
    def __init__(self, input_shape, output_dim):
        self.input_shape = input_shape
        self.output_dim = output_dim

    def build(self, hp):
        model = Sequential([
            TCN(input_shape=self.input_shape,
                nb_filters=hp.Int('nb_filters', min_value=32, max_value=512, step=32),
                kernel_size=hp.Choice('kernel_size', values=[2, 3, 6, 8]),
                padding='same',
                dilations=[1, 2, 4, 8]),
            Dense(self.output_dim)
        ])
        model.compile(optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                      loss='mean_squared_error')
        return model

# Model training and tuning for CPU
cpu_hypermodel = TCNHyperModel(input_shape=(training_features_reshaped.shape[1], 1), output_dim=1)
cpu_tuner = RandomSearch(cpu_hypermodel, objective='val_loss', max_trials=10, executions_per_trial=1, directory='my_dir', project_name='tcn_cpu')
cpu_tuner.search(training_features_reshaped, training_cpu_labels, epochs=20, validation_split=0.1)
best_cpu_hps = cpu_tuner.get_best_hyperparameters(num_trials=1)[0]
cpu_model = cpu_tuner.hypermodel.build(best_cpu_hps)
cpu_model.fit(training_features_reshaped, training_cpu_labels, epochs=100, validation_split=0.1)

# Model training and tuning for Memory
memory_hypermodel = TCNHyperModel(input_shape=(training_features_reshaped.shape[1], 1), output_dim=1)
memory_tuner = RandomSearch(memory_hypermodel, objective='val_loss', max_trials=10, executions_per_trial=1, directory='my_dir', project_name='tcn_memory')
memory_tuner.search(training_features_reshaped, training_memory_labels, epochs=20, validation_split=0.1)
best_memory_hps = memory_tuner.get_best_hyperparameters(num_trials=1)[0]
memory_model = memory_tuner.hypermodel.build(best_memory_hps)
memory_model.fit(training_features_reshaped, training_memory_labels, epochs=100, validation_split=0.1)

# Evaluation for CPU model
cpu_test_loss = cpu_model.evaluate(test_features_reshaped, test_cpu_labels, verbose=1)
print("CPU Test Loss:", cpu_test_loss)

# Evaluation for Memory model
memory_test_loss = memory_model.evaluate(test_features_reshaped, test_memory_labels, verbose=1)
print("Memory Test Loss:", memory_test_loss)

# Predictions
cpu_predictions = cpu_model.predict(test_features_reshaped)
memory_predictions = memory_model.predict(test_features_reshaped)

# Ensure predictions are non-negative
cpu_predictions = np.maximum(cpu_predictions, 0)
memory_predictions = np.maximum(memory_predictions, 0)

# Plotting for CPU Usage
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(cpu_predictions, label='Predicted CPU Usage')
plt.plot(test_cpu_labels, label='Real CPU Usage', linestyle='--')
plt.title('CPU Usage')
plt.legend()

# Plotting for Memory Usage
plt.subplot(1, 2, 2)
plt.plot(memory_predictions, label='Predicted Memory Usage')
plt.plot(test_memory_labels, label='Real Memory Usage', linestyle='--')
plt.title('Memory Usage')
plt.legend()

plt.tight_layout()
plt.show()
