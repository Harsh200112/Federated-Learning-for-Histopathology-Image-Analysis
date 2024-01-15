import torch
import torch.nn as nn
import torchmetrics
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import datasets

device = "cpu"

# roc_score = torchmetrics.classification.ROC(num_classes=2, task='binary')
f1_score_metric = torchmetrics.classification.BinaryF1Score()


class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(in_features=16, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


# Hyperparameters
image_size = 28
batch_size = 32
clr = 0.001
c_epochs = 10
num_clients = 3
mu = 0.05

transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder('Breast Cancer', transform=transforms)

classes = dataset.classes


def train_test_ds(data, test_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(data))), test_size=test_split)
    train_data = Subset(data, train_idx)
    test_data = Subset(data, val_idx)

    return train_data, test_data


train_data, x_data = train_test_ds(dataset)
test_data, val_data = train_test_ds(x_data, 0.5)

# DataLoaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)


def get_accuracy(model, data):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in data:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

            # roc_score.update(scores, y)
            f1_score_metric.update(predictions, y)

    return round((float(num_correct) / float(num_samples)) * 100, 3)


model = CNN(3, 2).to(device)
optimizer = optim.SGD(model.parameters(), lr=clr)
criterion = nn.CrossEntropyLoss()

for epoch in range(c_epochs):
    model.train()
    running_loss = 0.0
    for data, targets in train_loader:
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # forward Prop
        scores = model(data)
        loss = criterion(scores, targets)

        # backward prop
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    val_acc = get_accuracy(model, val_loader)
    print("Epoch:-", epoch+1, "| Val Accuracy:-", val_acc, f"| Training Loss: {running_loss / len(train_loader)}")

print("Test Accuracy =", get_accuracy(model, test_loader))


# roc_score_result = roc_score.compute()
f1_score_result = f1_score_metric.compute()
# print("ROC Report:", roc_score_result)
print("F1 Score:", f1_score_result)
