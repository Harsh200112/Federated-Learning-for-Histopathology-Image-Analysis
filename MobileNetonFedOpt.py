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
    project='FedProx on Breast Cancer Dataset (MobileNet)',
    config={
        "learning_rate": 0.01,
        "client_learning_rate": 0.001,
        "architecture": "FedProx",
        "dataset": "Breast Cancer",
        "client_epochs": 20,
        "communications": 20
    }
)

# Hyperparameters
image_size = 224
batch_size = 32
clr = 3e-4
c_epochs = 10
num_clients = 3
mu = 0.01

transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder('Overall-Dataset', transform=transforms)

classes = dataset.classes


# Model
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
            ConvBlock(3, 32, 2),
            InvertedResidual(32, 16, 1, 1),
            InvertedResidual(16, 24, 2, 3),
            ConvBlock(24, 128, 2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

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


def train_test_ds(data, test_split):
    train_idx, val_idx = train_test_split(list(range(len(data))), test_size=test_split)
    train_data = Subset(data, train_idx)
    test_data = Subset(data, val_idx)

    return train_data, test_data


train_data, x_data = train_test_ds(dataset, test_split=0.3)
test_data, val_data = train_test_ds(x_data, 1/3)

# DataLoaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)

# Client Data Loaders
c_train_loaders = []
data_loader_size = len(train_data) // num_clients

for i in range(num_clients):
    start_idx = i * data_loader_size
    end_idx = (i + 1) * data_loader_size if i < num_clients - 1 else len(train_data)

    subset = Subset(train_data, list(range(start_idx, end_idx)))
    loader = DataLoader(subset, batch_size=32, shuffle=True)

    c_train_loaders.append(loader)


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
        optimizer = optim.SGD(model.parameters(), lr=clr, momentum=0.9)
        optimizers.update({optim_name: optimizer})

        criterion_name = "criterion" + str(i)
        criterion = nn.CrossEntropyLoss()
        criterions.update({criterion_name: criterion})

    return models, optimizers, criterions


def get_val_loss(model, criterion, data, server):
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for x, y in data:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            val_loss += criterion(scores, y)

    return val_loss.item() / len(data)


# Setting the Main Model Parameters
def update_centralized_model(cent_model, models, num_clients):
    target_state_dict = cent_model.state_dict()
    temp_state_dict = cent_model.state_dict()

    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            target_state_dict[key].data.fill_(0.)
            for i in range(num_clients):
                model_name = "model" + str(i)
                model = models[model_name]
                state_dict = model.state_dict()
                target_state_dict[key].data += state_dict[key].data.clone() / num_clients
                if i==0:
                    target_state_dict[key].grad = (temp_state_dict[key].data.clone() - state_dict[key].data.clone()) / num_clients
                else:
                    target_state_dict[key].grad += (temp_state_dict[key].data.clone() - state_dict[key].data.clone()) / num_clients

    cent_model.load_state_dict(target_state_dict)
    return cent_model


traini_loss = []
valida_loss = []
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

                # Backward Prop
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            val_loss = get_val_loss(model, criterion, val_loader, server)

            wandb.log({"Training Loss": running_loss / len(c_train_loaders[i]), "Validation Loss": val_loss})


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
cent_lr = 0.001
cent_model = MobileNetV2(2).to(device)
cent_model = update_centralized_model(cent_model, models, num_clients)
cent_optimizer = optim.Adagrad(cent_model.parameters(), lr=cent_lr)
num_communications = 9

print("---Centralized Model---")
acc, f1_sc, auroc = get_accuracy(cent_model, test_loader)
wandb.log({"Server Model Accuracy": acc})
print("Communication", 1, "| Test Accuracy =", acc, "| F1-Score =", f1_sc, "| Auroc Score =", auroc)

for i in range(num_communications):
    cent_model.train()
    models = update_client_models(cent_model, models, num_clients)
    train_clients(num_clients, cent_model, models, optimizers, criterions)
    cent_optimizer.zero_grad()
    cent_model = update_centralized_model(cent_model, models, num_clients)
    cent_optimizer.step()
    acc, f1_sc, auroc = get_accuracy(cent_model, test_loader)
    wandb.log({"Server Model Accuracy": acc})
    print("Communication", i + 2, "| Test Accuracy =", acc, "| F1-Score =", f1_sc, "| Auroc Score =", auroc)

torch.save(cent_model, 'Models/Cent_Model.pth')

print("---Client Models---")
for i in range(num_clients):
    model_name = "model" + str(i)
    filepath = "Models/Client_Model-" + str(i + 1) + ".pth"
    acc, f1_sc, auroc = get_accuracy(models[model_name], test_loader)
    print(f"Test Accuracy for Client-{i + 1} is:-", acc)
    print(f"F1-Score for Client-{i + 1} is:-", f1_sc)
    print(f"Auroc Score for Client-{i + 1} is:-", auroc)
    torch.save(models[model_name], filepath)

wandb.finish()
