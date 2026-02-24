import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1): # El constructor de la clase LSTM define los parámetros del modelo, incluyendo el tamaño de entrada, el tamaño oculto, el número de capas y el tamaño de salida. Se inicializan las capas LSTM y la capa completamente conectada (fully connected) para la predicción final.
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM( 
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        ) 

        self.fc = nn.Linear(hidden_size, output_size) # La capa completamente conectada toma la salida del último estado oculto de la LSTM y la transforma en la predicción final del modelo。

    # El método forward define cómo se propagan los datos a través del modelo. La salida de la LSTM 
    # se toma solo para el último paso de tiempo (out[:, -1, :]) y se pasa a través de la capa 
    # completamente conectada para obtener la predicción final。
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out