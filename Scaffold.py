# Libraries
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

wandb.login(key='f4d9abb112afd97ffb569d0502533553957c14d8')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

wandb.init(
    project='SCAFFOLD',
    config={
        "client_learning_rate": 0.001,
        "server_learning_rate": 0.01,
        "architecture": "SCAFFOLD",
        "dataset": "MNIST, CIFAR10",
        "epochs": 10
    }
)


# Sever Hyperparameters
cv = 0  # Server Variate
communications = 20

# Client Hyperparameters
c_epochs = 10
clr = 0.001
num_clients = 10
batch_size = 32
image_size = 28
cc = torch.zeros(num_clients, dtype=torch.float)  # Client Variates
cp = torch.zeros(num_clients, dtype=torch.float)  # Changed Variates

train_data = datasets.MNIST('dataset', download=True, train=True, transform=transforms.ToTensor())
test_data = datasets.MNIST('dataset', download=True, train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

c_train_loaders = []
data_loader_size = len(train_data) // num_clients

for i in range(num_clients):
    start_idx = i * data_loader_size
    end_idx = (i + 1) * data_loader_size if i < num_clients - 1 else len(train_data)

    subset = Subset(train_data, list(range(start_idx, end_idx)))
    loader = DataLoader(subset, batch_size=32, shuffle=True)

    c_train_loaders.append(loader)


# NN
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


def get_clients_optimizer_criterion(num_clients):
    models = dict()
    optimizers = dict()
    criterions = dict()

    for i in range(num_clients):
        model_name = "model" + str(i)
        optimizer_name = "optim" + str(i)
        criterion_name = "criterion" + str(i)

        model = NN(784, 10).to(device)
        models.update({model_name: model})

        optimizer = optim.SGD(model.parameters(), lr=clr)
        optimizers.update({optimizer_name: optimizer})

        criterion = nn.CrossEntropyLoss()
        criterions.update({criterion_name: criterion})

    return models, optimizers, criterions


models, optimizers, criterions = get_clients_optimizer_criterion(num_clients)


def train_client(num_clients, models, server):
    for i in range(num_clients):
        model = models["model" + str(i)]
        optimizer = optimizers["optim" + str(i)]
        criterion = criterions["criterion" + str(i)]

        for epoch in range(c_epochs):
            for batch_idx, (data, targets) in enumerate(c_train_loaders[i]):
                data = data.to(device)
                targets = targets.to(device)
                data = data.reshape(data.shape[0], -1)

                # Forward Prop
                scores = model(data)
                loss = criterion(scores, targets) - clr * (cv - cc[i])

                wandb.log({"Overall Loss": loss})

                # Back Prop
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

        param_diff = {}
        with torch.no_grad():
            for name, param in server.named_parameters():
                local_param = model.state_dict()[name]
                if param.shape == local_param.shape:
                    param_diff[name] = param - local_param

        model.train()
        # Updating Client Variate
        cp[i] = cc[i] - cv + (1 / (clr * num_clients)) * sum([torch.norm(diff) for diff in param_diff.values()])


def update_server(cent_model, models, num_clients):
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


def update_clients(models, num_clients, server):
    with torch.no_grad():
        for i in range(num_clients):
            for param1, param2 in zip(models["model"+str(i)].parameters(), server.parameters()):
                param1.data = param2.data
    return models


# Server Model
server = NN(784, 10).to(device)

for comm in range(communications):
    models = update_clients(models, num_clients, server)
    train_client(num_clients, models, server)
    server = update_server(models, num_clients, cv, server)
    digit_test_acc = get_accuracy(server, test_loader)
    wandb.log({"Server Digit Acc": digit_test_acc})
    print("Communication No", comm + 1, "| Digit Test Accuracy =", digit_test_acc)

wandb.finish()
