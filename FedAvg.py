# Libraries
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms
from torchvision import datasets

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(
    project='FedAvg',
    config={
        "learning_rate": 0.01,
        "architecture": "FedAvg",
        "dataset": "MNIST",
        "epochs": 10
    }
)

# Dataset
train_data = datasets.MNIST('dataset', train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST('dataset', train=False, transform=transforms.ToTensor(), download=True)


# Client Model
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Client Hyperparameters
clr = 0.01
c_epochs = 5
num_clients = 10

# Main_Model Hyperparameters
mlr = 0.001
m_batch_size = 32
m_epochs = 5

# Main Model i.e. Model without using Federated Learning
main_model = NN(784, 10)
main_model.to(device)

m_optimizer = optim.SGD(main_model.parameters(), lr=mlr)
m_criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(train_data, batch_size=m_batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=m_batch_size, shuffle=True)

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

    with torch.no_grad():
        for x, y in data:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    model.train()

    return round((float(num_correct) / float(num_samples)) * 100, 3)


# Training Main_Model
print("---Evaluating Main Model---")
for epoch in range(m_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        data = data.reshape(data.shape[0], -1)

        print(data.shape)

        # forward
        scores = main_model(data)
        loss = m_criterion(scores, targets)

        wandb.log({"loss": loss})

        # backward
        m_optimizer.zero_grad()
        loss.backward()

        m_optimizer.step()

    print("Epoch No", epoch, "| Train Accuracy =", get_accuracy(main_model, train_loader), "| Test Accuracy =",
          get_accuracy(main_model, test_loader))
print('\n')


# Creating Different Client Models
def get_client_models(num_clients):
    models = dict()
    optimizers = dict()
    criterions = dict()

    for i in range(num_clients):
        modelName = "model" + str(i)
        model = NN(784, 10)
        models.update({modelName: model})

        optim_name = "optim" + str(i)
        optimizer = optim.SGD(model.parameters(), lr=clr)
        optimizers.update({optim_name: optimizer})

        criterion_name = "criterion" + str(i)
        criterion = nn.CrossEntropyLoss()
        criterions.update({criterion_name: criterion})

    return models, optimizers, criterions


# Getting the Average Weight
def get_average_weights(models, num_clients):
    fc1_mean_weight = torch.zeros(models["model0"].fc1.weight.shape)
    fc1_mean_bias = torch.zeros(models["model0"].fc1.bias.shape)

    fc2_mean_weight = torch.zeros(models["model0"].fc2.weight.shape)
    fc2_mean_bias = torch.zeros(models["model0"].fc2.bias.shape)

    with torch.no_grad():
        for i in range(num_clients):
            model_name = "model" + str(i)
            fc1_mean_weight += models[model_name].fc1.weight.data.clone()
            fc1_mean_bias += models[model_name].fc1.bias.data.clone()

            fc2_mean_weight += models[model_name].fc2.weight.data.clone()
            fc2_mean_bias += models[model_name].fc2.bias.data.clone()

        fc1_mean_weight = fc1_mean_weight / num_clients
        fc1_mean_bias = fc1_mean_bias / num_clients
        fc2_mean_weight = fc2_mean_weight / num_clients
        fc2_mean_bias = fc2_mean_bias / num_clients

    return fc1_mean_weight, fc1_mean_bias, fc2_mean_weight, fc2_mean_bias


# Setting the Main Model Parameters
def set_main_model_parameters(centralized_model, models, num_clients):
    fc1_mean_weight, fc1_mean_bias, fc2_mean_weight, fc2_mean_bias = get_average_weights(models, num_clients)

    with torch.no_grad():
        centralized_model.fc1.weight.data = fc1_mean_weight.data.clone()
        centralized_model.fc1.bias.data = fc1_mean_bias.data.clone()

        centralized_model.fc2.weight.data = fc2_mean_weight.data.clone()
        centralized_model.fc2.bias.data = fc2_mean_bias.data.clone()

    return centralized_model


# Training Clients
models, optimizers, criterions = get_client_models(num_clients)


def train_clients(num_clients):
    for i in range(num_clients):
        model_name = "model" + str(i)
        optimizer_name = "optim" + str(i)
        criterion_name = "criterion" + str(i)
        model = models[model_name]
        optimizer = optimizers[optimizer_name]
        criterion = criterions[criterion_name]

        for epoch in range(c_epochs):
            for batch_idx, (data, targets) in enumerate(c_train_loaders[i]):
                data = data.to(device)
                targets = targets.to(device)

                print(data.shape)

                data = data.reshape(data.shape[0], -1)

                # forward Prop
                scores = model(data)
                loss = criterion(scores, targets)

                wandb.log({"loss": loss})

                # backward prop
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()


# Updating Client Models
def update_client_models(cent_model, models, num_clients):
    with torch.no_grad():
        for i in range(num_clients):
            model_name = "model" + str(i)
            models[model_name].fc1.weight.data = cent_model.fc1.weight.data.clone()
            models[model_name].fc1.bias.data = cent_model.fc1.bias.data.clone()

            models[model_name].fc2.weight.data = cent_model.fc2.weight.data.clone()
            models[model_name].fc2.bias.data = cent_model.fc2.bias.data.clone()

    return models


# Centralized Model
cent_model = NN(784, 10).to(device)
cent_model = set_main_model_parameters(cent_model, models, num_clients)

print("---Centralized Model---")
print("Iteration 1", "| Test Accuracy =", get_accuracy(cent_model, train_loader))

for i in range(9):
    models = update_client_models(cent_model, models, num_clients)
    train_clients(num_clients)
    cent_model = set_main_model_parameters(cent_model, models, num_clients)
    acc = get_accuracy(cent_model, train_loader)
    wandb.log({"acc": acc})
    print("Iteration", i + 2, "| Test Accuracy =", acc)
