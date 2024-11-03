import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from tqdm import tqdm
from torcheval.metrics import MulticlassAccuracy
from matplotlib import pyplot as plt
from dataset import X_train, y_train, X_valid, y_valid, y_valid_for_accuracy

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten()

        self.dense1 = nn.Linear(128*16*53, 128)
        self.act1 = nn.ReLU()

        self.dense2= nn.Linear(128, 19)
        self.act2 = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.act1(x)
        x = self.dense2(x)
        x = self.act2(x)
        return x

def train(model, X_train, y_train, X_valid, y_valid, loss_function, optimizer, epoch):
    train_losses = []
    val_losses = []

    for _ in tqdm(range(epoch)):
        y_pred = model(X_train)
        loss = loss_function(y_pred, y_train)
        train_losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            y_pred = model(X_valid)
            val_loss = loss_function(y_pred, y_valid)
            val_losses.append(val_loss.item())
    
    return model, train_losses, val_losses

def accuracy(model, X_valid, y_valid_for_accuracy):
    with torch.no_grad():
        y_pred = model(X_valid)
        metric = MulticlassAccuracy()
        metric.update(y_pred, y_valid_for_accuracy)
        return metric.compute()


model = Model()

loss_function = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
epoch = 10

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {pytorch_total_params}") # 13 988 883
# pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

model, train_losses, val_losses = train(model, X_train, y_train, X_valid, y_valid, loss_function, optimizer, epoch)
pickle.dump(model, open('genre_guru_model.sav', 'wb'))

# model = pickle.load(open('genre_guru_model.sav', 'rb'))

accuracy_score = accuracy(model, X_valid, y_valid_for_accuracy)
print(f"Accuracy score: {accuracy_score}")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train loss', color='red')
plt.plot(val_losses, label='Validation loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss value')
plt.legend()
plt.show()



