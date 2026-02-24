import torch.optim as optim
import torch.nn as nn

def train_model_mlp(model, X_train, y_train, epochs=200, lr=0.001):

    criterion = nn.MSELoss() #Funcion de perdida MSE
    optimizer = optim.Adam(model.parameters(), lr=lr) #optimizador adam

    model.train()

    for epoch in range(epochs):

        optimizer.zero_grad() # Resetea los gradientes de todos los parametros del modelo

        outputs = model(X_train) # Se calcula la salida dada una entrada de datos de la RN
        loss = criterion(outputs, y_train) # Se calcula el error

        loss.backward() # Se realiza la retropropagacion
        optimizer.step() # Actualiza los parametros del modelo

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

    return model
