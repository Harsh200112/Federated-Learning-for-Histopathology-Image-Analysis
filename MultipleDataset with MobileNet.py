import torch
import wandb
import torch.nn as nn
import torchmetrics
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import datasets

wandb.login(key='f4d9abb112afd97ffb569d0502533553957c14d8')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

wandb.init(
    project='FedProx on Breast Cancer Dataset (Bracs, Bach and Break-His)',
    config={
        "learning_rate": 0.01,
        "client_learning_rate": 0.003,
        "architecture": "FedProx",
        "dataset": "Breast Cancer",
        "client_epochs": 15,
        "communications": 10
    }
)

# Hyperparameters
image_size = 40
batch_size = 32
clr = 3e-4
c_epochs = 15
num_clients = 3
mu = 0.01

train_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

breakhis_dataset = datasets.ImageFolder('Breast Cancer', transform=train_transforms)
bracs_dataset = datasets.ImageFolder('Bracs', transform=train_transforms)
bach_dataset = datasets.ImageFolder('ICIAR-BACH', transform=train_transforms)


# Model
class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.fc1 = nn.Linear(in_features=32, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def train_test_ds(data, test_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(data))), test_size=test_split)
    train_data = Subset(data, train_idx)
    test_data = Subset(data, val_idx)

    return train_data, test_data


breakhis_train_data, breakhis_test_data = train_test_ds(breakhis_dataset)
bracs_train_data, bracs_test_data = train_test_ds(bracs_dataset)
bach_train_data, bach_test_data = train_test_ds(bach_dataset)

# DataLoaders
breakhis_train_loader = DataLoader(breakhis_train_data, batch_size=batch_size, shuffle=True)
breakhis_test_loader = DataLoader(breakhis_test_data, batch_size=batch_size)

bracs_train_loader = DataLoader(bracs_train_data, batch_size=batch_size, shuffle=True)
bracs_test_loader = DataLoader(bracs_test_data, batch_size=batch_size)

bach_train_loader = DataLoader(bach_train_data, batch_size=batch_size, shuffle=True)
bach_test_loader = DataLoader(bach_test_data, batch_size=batch_size)

# Client Data Loaders
c_train_loaders = [breakhis_train_loader, bracs_train_loader, bach_train_loader]


def get_val_loss(model, criterion, data, server):
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for x, y in data:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            val_loss += criterion(scores, y)

            for param1, param2 in zip(model.parameters(), server.parameters()):
                val_loss += (mu / 2) * torch.norm((param1.data - param2.data), p=2)

    return val_loss.item() / len(data)


def get_accuracy(model, data):
    num_correct = 0
    num_samples = 0
    model.eval()

    f1_score = torchmetrics.classification.BinaryF1Score().to(device)
    auroc_score = torchmetrics.classification.BinaryAUROC().to(device)

    with torch.no_grad():
        for x, y in data:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

            f1_score.update(predictions, y)
            auroc_score.update(predictions, y)

        f1_score_result = f1_score.compute()
        auroc_score_result = auroc_score.compute()
    return round((float(num_correct) / float(num_samples)) * 100, 2), round(f1_score_result.item(), 2), round(
        auroc_score_result.item(), 2)


# Creating Different Client Models
def get_client_models(num_clients):
    models = dict()
    optimizers = dict()
    criterions = dict()

    for i in range(num_clients):
        modelName = "model" + str(i)
        model = CNN(3, 2).to(device)
        models.update({modelName: model})

        optim_name = "optim" + str(i)
        optimizer = optim.Adam(model.parameters(), lr=clr)
        optimizers.update({optim_name: optimizer})

        criterion_name = "criterion" + str(i)
        criterion = nn.CrossEntropyLoss()
        criterions.update({criterion_name: criterion})

    return models, optimizers, criterions


# Setting the Main Model Parameters
def update_centralized_model(cent_model, models, num_clients):
    target_state_dict = cent_model.state_dict()

    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            target_state_dict[key].data.fill_(0.)
            for i in range(num_clients):
                model_name = "model" + str(i)
                model = models[model_name]
                state_dict = model.state_dict()
                target_state_dict[key].data += state_dict[key].data.clone()/num_clients

    cent_model.load_state_dict(target_state_dict)
    return cent_model


# Training Clients
def train_clients(num_clients, server, models, optimizers, criterions):
    for i in range(num_clients):
        model_name = "model" + str(i)
        optimizer_name = "optim" + str(i)
        criterion_name = "criterion" + str(i)
        model = models[model_name]
        optimizer = optimizers[optimizer_name]
        criterion = criterions[criterion_name]

        for epoch in range(c_epochs):
            model.train()
            running_loss = 0.0
            for data, targets in c_train_loaders[i]:
                data = data.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                # Forward Prop
                scores = model(data)
                loss = criterion(scores, targets)

                # Adding the FedProx regularization term (L-2 Norm)
                for param1, param2 in zip(model.parameters(), server.parameters()):
                    loss += (mu / 2) * torch.norm((param1.data - param2.data), p=2)

                # Backward Prop
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            wandb.log({"Training Loss": running_loss / len(c_train_loaders[i])})


# Updating Client Models
def update_client_models(cent_model, models, num_clients):
    with torch.no_grad():
        for i in range(num_clients):
            model_name = "model" + str(i)
            for param1, param2 in zip(models[model_name].parameters(), cent_model.parameters()):
                param1.data = param2.data.clone()

    return models


# Client Models
models, optimizers, criterions = get_client_models(num_clients)

# Centralized Model
cent_model = CNN(3, 2).to(device)
cent_model = update_centralized_model(cent_model, models, num_clients)
num_communications = 9

print("---Centralized Model---")
acc1, f1_sc1, auroc1 = get_accuracy(cent_model, breakhis_test_loader)
acc2, f1_sc2, auroc2 = get_accuracy(cent_model, bracs_test_loader)
acc3, f1_sc3, auroc3 = get_accuracy(cent_model, bach_test_loader)
wandb.log({"Server Model Accuracy (break-his)": acc1, "Server Model Accuracy (bracs)": acc2,
           "Server Model Accuracy (bach)": acc3})
print("Communication", 1, "| Test Accuracy (break-his) =", acc1, "| Test Accuracy (bracs) =", acc2,
      "| Test Accuracy (bach) =", acc3,
      "| F1-Score (break-his) =", f1_sc1, "| F1-Score (bracs) =", f1_sc2, "| F1-Score (bach) =", f1_sc3,
      "| Auroc Score (break-his) =", auroc1, "| Auroc Score (bracs) =", auroc2, "| Auroc Score (bach) =", auroc3)

for i in range(num_communications):
    models = update_client_models(cent_model, models, num_clients)
    train_clients(num_clients, cent_model, models, optimizers, criterions)
    cent_model = update_centralized_model(cent_model, models, num_clients)
    acc1, f1_sc1, auroc1 = get_accuracy(cent_model, breakhis_test_loader)
    acc2, f1_sc2, auroc2 = get_accuracy(cent_model, bracs_test_loader)
    acc3, f1_sc3, auroc3 = get_accuracy(cent_model, bach_test_loader)
    wandb.log({"Server Model Accuracy (break-his)": acc1, "Server Model Accuracy (bracs)": acc2,
               "Server Model Accuracy (bach)": acc3})
    print("Communication", 1, "| Test Accuracy (break-his) =", acc1, "| Test Accuracy (bracs) =", acc2,
          "| Test Accuracy (bach) =", acc3,
          "| F1-Score (break-his) =", f1_sc1, "| F1-Score (bracs) =", f1_sc2, "| F1-Score (bach) =", f1_sc3,
          "| Auroc Score (break-his) =", auroc1, "| Auroc Score (bracs) =", auroc2, "| Auroc Score (bach) =", auroc3)

update_client_models(cent_model, models, num_clients)

torch.save(cent_model, 'Models/Cent_Model.pth')

# print("---Client Models---")
# for i in range(num_clients):
#     model_name = "model" + str(i)
#     filepath = "Models/Client_Model-" + str(i + 1) + ".pth"
#     acc1, f1_sc1 = get_accuracy(cent_model, breakhis_test_loader)
#     acc2, f1_sc2 = get_accuracy(cent_model, bracs_test_loader)
#     print(f"Test Accuracy for Client-{i + 1} is:-", acc1, "on (break-his) and", acc2, "on (bracs).")
#     print(f"F1-Score for Client-{i + 1} is:-", f1_sc1.item(), "on (break-his) and", f1_sc2.item(), "on (bracs).")
#     torch.save(models[model_name], filepath)
#
# wandb.finish()
