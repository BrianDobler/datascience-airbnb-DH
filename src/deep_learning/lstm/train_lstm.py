import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def train_model_lstm(model, X_train, y_train, 
                     epochs=50, lr=0.001, batch_size=16):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):

        model.train()
        epoch_loss = 0

        for xb, yb in loader:

            outputs = model(xb)
            loss = criterion(outputs, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(loader):.6f}")

    return model