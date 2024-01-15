# Libraries
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision import transforms

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(
    project='FedOPT',
    config={
        "learning_rate": 0.01,
        "architecture": "FedAvg",
        "dataset": "MNIST",
        "epochs": 10
    }
)

# Client Learning Hyperparameters
num_clients = 10
clr = 0.01
c_epochs = 10

# Main Model Hyperparameters
mlr = 0.001
batch_size = 32
m_epochs = 10

# Datasets
train_data = datasets.MNIST('dataset', train=True, transform=transforms.ToTensor())
test_data = datasets.MNIST('dataset', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

c_train_loaders = []
size_loader = len(train_loader) // num_clients
for i in range(num_clients):
    start_idx = i * size_loader
    end_idx = (i + 1) * size_loader if i < num_clients - 1 else len(train_data)

    subset = Subset(train_data, list(range(start_idx, end_idx)))
    loader = DataLoader(subset, batch_size=32, shuffle=True)

    c_train_loaders.append(loader)


# Neural Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Main Model
main_model = NN(784, 10)
main_model.to(device)

m_optimizer = optim.SGD(main_model.parameters(), lr=mlr)
m_criterion = nn.CrossEntropyLoss()


def get_accuracy(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return round((float(num_correct) / float(num_samples)) * 100, 3)


# Training Main Model
print("---Main Model Training---")
for epoch in range(m_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        data = data.reshape(data.shape[0], -1)  # Flatten

        # Forward Prop
        scores = main_model(data)
        loss = m_criterion(scores, targets)

        wandb.log({"main_model_loss": loss})

        # Backward Prop
        m_optimizer.zero_grad()
        loss.backward()

        m_optimizer.step()

    train_acc = get_accuracy(main_model, train_loader)
    test_acc = get_accuracy(main_model, test_loader)
    wandb.log({"main_model_train_acc": train_acc, "main_model_test_acc": test_acc})

    print("Epoch No", epoch + 1, "| Train Accuracy =", train_acc, "| Test Accuracy =",
          test_acc)

print('\n')


# Setting Up Clients
def get_client_models_optimizer_criterion(num_clients):
    models = dict()
    optimizers = dict()
    criterions = dict()

    for i in range(num_clients):
        model_name = "model" + str(i)
        optimizer_name = "optim" + str(i)
        criterion_name = "criterion" + str(i)

        model = NN(784, 10)
        model = model.to(device)
        models.update({model_name: model})

        criterion = nn.CrossEntropyLoss()
        criterions.update({criterion_name: criterion})

        optimizer = optim.SGD(model.parameters(), lr=clr)
        optimizers.update({optimizer_name: optimizer})

    return models, optimizers, criterions


# Average Gradient and Biases of Clients
def get_averaged_weights_biases(models, num_clients):
    fc1_mean_weights = torch.zeros(models["model0"].fc1.weight.shape)
    fc1_mean_bias = torch.zeros(models["model0"].fc1.bias.shape)

    fc2_mean_weights = torch.zeros(models["model0"].fc2.weight.shape)
    fc2_mean_bias = torch.zeros(models["model0"].fc2.bias.shape)

    fc3_mean_weights = torch.zeros(models["model0"].fc3.weight.shape)
    fc3_mean_bias = torch.zeros(models["model0"].fc3.bias.shape)

    with torch.no_grad():
        for i in range(num_clients):
            model_name = "model" + str(i)
            fc1_mean_weights += models[model_name].fc1.weight.data.clone()
            fc1_mean_bias += models[model_name].fc1.bias.data.clone()

            fc2_mean_weights += models[model_name].fc2.weight.data.clone()
            fc2_mean_bias += models[model_name].fc2.bias.data.clone()

            fc3_mean_weights += models[model_name].fc3.weight.data.clone()
            fc3_mean_bias += models[model_name].fc3.bias.data.clone()

    fc1_mean_weights = fc1_mean_weights / num_clients
    fc1_mean_bias = fc1_mean_bias / num_clients
    fc2_mean_weights = fc2_mean_weights / num_clients
    fc2_mean_bias = fc2_mean_bias / num_clients
    fc3_mean_weights = fc3_mean_weights / num_clients
    fc3_mean_bias = fc3_mean_bias / num_clients

    return fc1_mean_weights, fc1_mean_bias, fc2_mean_weights, fc2_mean_bias, fc3_mean_weights, fc3_mean_bias


# Setting Centralized Models Parameters
def set_centralized_model_parameters(centralized_model, models, num_clients):
    fc1_mean_weights, fc1_mean_bias, fc2_mean_weights, fc2_mean_bias, fc3_mean_weights, fc3_mean_bias = get_averaged_weights_biases(
        models, num_clients)

    with torch.no_grad():
        centralized_model.fc1.weight.data = fc1_mean_weights.data.clone()
        centralized_model.fc1.bias.data = fc1_mean_bias.data.clone()

        centralized_model.fc2.weight.data = fc2_mean_weights.data.clone()
        centralized_model.fc2.bias.data = fc2_mean_bias.data.clone()

        centralized_model.fc3.weight.data = fc3_mean_weights.data.clone()
        centralized_model.fc3.bias.data = fc3_mean_bias.data.clone()

    return centralized_model


models, optimizers, criterions = get_client_models_optimizer_criterion(num_clients)


def train_client_models(num_clients):
    for i in range(num_clients):
        model_name = "model" + str(i)
        optimizer_name = "optim" + str(i)
        criterion_name = "criterion" + str(i)

        model = models[model_name]
        optimizer = optimizers[optimizer_name]
        criterion = criterions[criterion_name]

        model.train()

        for epoch in range(c_epochs):
            for batch_idx, (data, targets) in enumerate(c_train_loaders[i]):
                data = data.to(device)
                targets = targets.to(device)
                data = data.reshape(data.shape[0], -1)

                # forward
                scores = model(data)
                loss = criterion(scores, targets)

                wandb.log({"client_model_loss": loss})

                # backward
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()


def update_clients(centralized_model, models, num_clients):
    with torch.no_grad():
        for i in range(num_clients):
            model_name = "model" + str(i)
            model = models[model_name]

            model.fc1.weight.data = centralized_model.fc1.weight.data.clone()
            model.fc1.bias.data = centralized_model.fc1.bias.data.clone()

            model.fc2.weight.data = centralized_model.fc2.weight.data.clone()
            model.fc2.bias.data = centralized_model.fc2.bias.data.clone()

            model.fc3.weight.data = centralized_model.fc3.weight.data.clone()
            model.fc3.bias.data = centralized_model.fc3.bias.data.clone()

    return models


# Centralized Model
clr = 0.01
centralized_model = NN(784, 10)
centralized_model.to(device)
centralized_model = set_centralized_model_parameters(centralized_model, models, num_clients)
cent_optimizer = optim.Adagrad(centralized_model.parameters(), lr=clr)

print("---Centralized Model---")
print("Iteration 1", "| Test Accuracy =", get_accuracy(centralized_model, test_loader))

for i in range(14):
    models = update_clients(centralized_model, models, num_clients)
    train_client_models(num_clients)
    with torch.no_grad():
        for param_centralized in centralized_model.parameters():
            param_centralized.zero_()
        centralized_model = set_centralized_model_parameters(centralized_model, models, num_clients)
    cent_optimizer.step()
    test_acc = get_accuracy(centralized_model, train_loader)
    wandb.log({"Centralized_Model_Test_Acc": test_acc})
    print("Iteration", i + 2, "| Test Accuracy =", test_acc)
