import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torcheval.metrics import MulticlassAccuracy
from tqdm import tqdm
from matplotlib import pyplot as plt
from dataset import device, train_loader, valid_loader


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flatten = nn.Flatten()

        self.dense1 = nn.Linear(164864, 128)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x


def train(model, train_loader, valid_loader, loss_function, optimizer, epochs, device):
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs)):
        model.train()
        train_losses_for_batch = []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)

            loss = loss_function(y_pred, y_batch)
            train_losses_for_batch.append(loss.item())

            loss.backward()
            optimizer.step()

        train_loss_mean = torch.mean(torch.tensor(train_losses_for_batch)).item()
        train_losses.append(train_loss_mean)

        model.eval()
        val_losses_for_batch = []

        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                y_pred = model(X_batch)
                val_loss = loss_function(y_pred, y_batch)
                val_losses_for_batch.append(val_loss.item())

        val_loss_mean = torch.mean(torch.tensor(val_losses_for_batch)).item()
        val_losses.append(val_loss_mean)

        print(
            f"Epoch {epoch + 1}/{epochs}, Train loss: {train_loss_mean:.4f}, Validation Loss: {val_loss_mean:.4f}"
        )

    return model, train_losses, val_losses


def accuracy(model, valid_loader, device):
    model.eval()
    metric = MulticlassAccuracy(num_classes=10)  
    with torch.no_grad():  
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            y_pred_prob = F.softmax(y_pred, dim=1)
            y_pred_classes = torch.argmax(y_pred_prob, dim=1)
            metric.update(y_pred_classes, y_batch)  

    return metric.compute()  


model = Model().to(device)

loss_function = nn.CrossEntropyLoss()
learning_rate = 0.001  # 2e-5, 2e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
epoch = 10

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {pytorch_total_params}")
# pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

model, train_losses, val_losses = train(
    model, train_loader, valid_loader, loss_function, optimizer, epoch, device
)

torch.save(model.state_dict(), "model.pth")

accuracy_score = accuracy(model, valid_loader, device)
print(f"Accuracy score: {accuracy_score}")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train loss", color="red")
plt.plot(val_losses, label="Validation loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss value")
plt.legend()
plt.show()


# model = Model() 
# model.load_state_dict(torch.load('model.pth'))
