import numpy as np

def create_sequences(data, seq_length):
    """
    Crea secuencias de datos para entrenamiento de modelos LSTM.
    
    Args:
        data (array-like): Serie temporal de datos.
        seq_length (int): Longitud de cada secuencia.
        
    Returns:
        X (numpy array): Matriz de características con forma (num_samples, seq_length, num_features).
        y (numpy array): Vector de etiquetas con forma (num_samples,).
    """
    X, y = [], [] # Inicializa listas vacías para almacenar las secuencias de características (X) y las etiquetas (y).
    for i in range(len(data) - seq_length): # Itera sobre la serie temporal, creando secuencias de longitud 'seq_length' y sus correspondientes etiquetas.
        X.append(data[i:i+seq_length]) 
        y.append(data[i+seq_length]) 
    return np.array(X), np.array(y)