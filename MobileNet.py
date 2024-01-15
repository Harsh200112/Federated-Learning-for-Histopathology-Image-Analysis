import torch
import wandb
import torch.nn as nn
import torchmetrics
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import datasets

wandb.login(key='f4d9abb112afd97ffb569d0502533553957c14d8')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

wandb.init(
    project='FedProx Using MobileNet on Breast Cancer Dataset (Bracs and Break-His)',
    config={
        "learning_rate": 0.01,
        "client_learning_rate": 0.001,
        "architecture": "FedProx",
        "dataset": "Breast Cancer",
        "client_epochs": 15,
        "communications": 10
    }
)

# Hyperparameters
image_size = 224
batch_size = 64
clr = 3e-4
c_epochs = [15, 15]
num_clients = 2
mu = 0.01

train_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

breakhis_dataset = datasets.ImageFolder('Breast Cancer', transform=train_transforms)
bracs_dataset = datasets.ImageFolder('Bracs', transform=train_transforms)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetV2, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 32, 2),  # Initial Convolutional Layer
            InvertedResidual(32, 16, 1, 1),
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 32, 3, 6),
            InvertedResidual(32, 64, 4, 6),
            InvertedResidual(64, 96, 3, 6),
            InvertedResidual(96, 160, 3, 6),
            InvertedResidual(160, 320, 1, 6),
            ConvBlock(320, 1280, 1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, t, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        # Define the layers
        layers = []
        if t != 1:
            layers.append(nn.Conv2d(in_channels, in_channels * t, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(in_channels * t))
            layers.append(nn.ReLU6(inplace=True))

        layers.extend([
            nn.Conv2d(in_channels * t, in_channels * t, 3, stride, 1, groups=in_channels * t, bias=False),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * t, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def train_test_ds(data, test_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(data))), test_size=test_split)
    train_data = Subset(data, train_idx)
    test_data = Subset(data, val_idx)

    return train_data, test_data


breakhis_train_data, breakhis_test_data = train_test_ds(breakhis_dataset)
bracs_train_data, bracs_test_data = train_test_ds(bracs_dataset)

# DataLoaders
breakhis_train_loader = DataLoader(breakhis_train_data, batch_size=batch_size, shuffle=True)
breakhis_test_loader = DataLoader(breakhis_test_data, batch_size=batch_size)

bracs_train_loader = DataLoader(bracs_train_data, batch_size=batch_size, shuffle=True)
bracs_test_loader = DataLoader(bracs_test_data, batch_size=batch_size)

# Client Data Loaders
c_train_loaders = [breakhis_train_loader, bracs_train_loader]


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
        model = MobileNetV2(2).to(device)
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

        for epoch in range(c_epochs[i]):
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
cent_model = MobileNetV2(2).to(device)
cent_model = update_centralized_model(cent_model, models, num_clients)
num_communications = 9

print("---Centralized Model---")
acc1, f1_sc1, auroc1 = get_accuracy(cent_model, breakhis_test_loader)
acc2, f1_sc2, auroc2 = get_accuracy(cent_model, bracs_test_loader)
wandb.log({"Server Model Accuracy (break-his)": acc1})
wandb.log({"Server Model Accuracy (bracs)": acc2})
print("Communication", 1, "| Test Accuracy (break-his) =", acc1, "| Test Accuracy (bracs) =", acc2,
      "| F1-Score (break-his) =", f1_sc1, "| F1-Score (bracs) =", f1_sc2, "| Auroc Score (Breakhis) =", auroc1,
      "| Auroc Score (Bracs) =", auroc2)

for i in range(num_communications):
    models = update_client_models(cent_model, models, num_clients)
    train_clients(num_clients, cent_model, models, optimizers, criterions)
    cent_model = update_centralized_model(cent_model, models, num_clients)
    acc1, f1_sc1, auroc1 = get_accuracy(cent_model, breakhis_test_loader)
    acc2, f1_sc2, auroc2 = get_accuracy(cent_model, bracs_test_loader)
    wandb.log({"Server Model Accuracy (break-his)": acc1})
    wandb.log({"Server Model Accuracy (bracs)": acc2})
    print("Communication", i+2, "| Test Accuracy (break-his) =", acc1, "| Test Accuracy (bracs) =", acc2,
          "| F1-Score (break-his) =", f1_sc1, "| F1-Score (bracs) =", f1_sc2, "| Auroc Score (Breakhis) =", auroc1,
          "| Auroc Score (Bracs) =", auroc2)

update_client_models(cent_model, models, num_clients)

torch.save(cent_model, 'Models/Cent_Model.pth')

print("---Client Models---")
for i in range(num_clients):
    model_name = "model" + str(i)
    filepath = "Models/Client_Model-" + str(i + 1) + ".pth"
    acc1, f1_sc1, auroc1 = get_accuracy(models[model_name], breakhis_test_loader)
    acc2, f1_sc2, auroc2 = get_accuracy(models[model_name], bracs_test_loader)
    print(f"Test Accuracy for Client-{i + 1} is:-", acc1, "on (break-his) and", acc2, "on (bracs).")
    print(f"F1-Score for Client-{i + 1} is:-", f1_sc1, "on (break-his) and", f1_sc2, "on (bracs).")
    print(f"Auroc Score for Client-{i + 1} is:-", auroc1, "on (break-his) and", auroc2, "on (bracs).")
    torch.save(models[model_name], filepath)

wandb.finish()
