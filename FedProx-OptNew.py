# Importing Libraries
import torch
import torch.nn as nn
import torchmetrics
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import datasets

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Hyperparameters
image_size = 224
batch_size = 32
clr = 3e-4
c_epochs = 10
num_clients = 3
mu = 0.01

# Preprocessing
transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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


# Train, Test, Validation Splitting
def train_test_ds(data, test_split=0.3):
    train_idx, val_idx = train_test_split(list(range(len(data))), test_size=test_split)
    train_data = Subset(data, train_idx)
    test_data = Subset(data, val_idx)

    return train_data, test_data


train_data, x_data = train_test_ds(dataset)
test_data, val_data = train_test_ds(x_data, 1/3)


class DataLoaderWrapper(DataLoader):
    def __iter__(self):
        return ((key.to(device), value.to(device)) for key, value in super().__iter__())


# DataLoaders
train_loader = DataLoaderWrapper(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoaderWrapper(test_data, batch_size=batch_size, pin_memory=True)
val_loader = DataLoaderWrapper(val_data, batch_size=batch_size, pin_memory=True)

# Client Data Loaders
c_train_loaders = []
data_loader_size = len(train_data) // num_clients

for i in range(num_clients):
    start_idx = i * data_loader_size
    end_idx = (i + 1) * data_loader_size if i < num_clients - 1 else len(train_data)

    subset = Subset(train_data, list(range(start_idx, end_idx)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    c_train_loaders.append(loader)


# Accuracy, Auroc, F1-Scre Calculation
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


# Validation Loss Calculation
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
                    target_state_dict[key].grad = (state_dict[key].data.clone() - temp_state_dict[key].data.clone()) / num_clients
                else:
                    target_state_dict[key].grad += (state_dict[key].data.clone() - temp_state_dict[key].data.clone()) / num_clients

    cent_model.load_state_dict(target_state_dict)
    return cent_model


# Finding the Rank of Layers important in predicting the final outcome
def get_ranked_layers(clients, num_clients, c_train_loaders):
    for no in range(num_clients):
        model_name = "model" + str(no)
        model = clients[model_name]
        layers = ['Conv Block-1', 'Inverted Residual Block-1 1st Conv', 'Inverted Residual Block-1 2nd Conv', 'Inverted Residual Block-2 1st Conv', 'Inverted Residual Block-2 2nd Conv', 'Inverted Residual Block-2 3rd Conv', 'Conv Block-2', 'Avg Pool']
        target_layers = [[model.model[0].conv[0]], [model.model[1].conv[0]], [model.model[1].conv[3]], [model.model[2].conv[0]], [model.model[2].conv[3]],
                         [model.model[2].conv[6]], [model.model[3].conv[0]], [model.avgpool]]
        ranked_layers = []

        print("----Ranked Layers For Client", no+1,"----")
        layer_no = 0
        for layer in target_layers:
            mse = 0
            for x, y in c_train_loaders[no]:
                x = x.to(device)
                y = y.to(device)
                for i in range(batch_size):
                    for l, k in c_train_loaders[no]:
                        l = l.to(device)
                        k = k.to(device)
                        for j in range(batch_size):
                            cam = GradCAM(model=model, target_layers=layer)
                            target1 = [ClassifierOutputTarget(y[i])]
                            target2 = [ClassifierOutputTarget(k[j])]

                            grayscale_cam1 = cam(input_tensor=x[i].unsqueeze(0), targets=target1)
                            grayscale_cam2 = cam(input_tensor=l[j].unsqueeze(0), targets=target2)

                            grayscale_cam1 = grayscale_cam1[0, :]
                            grayscale_cam2 = grayscale_cam2[0, :]

                            mse += 1/batch_size * ((grayscale_cam1 - grayscale_cam2) / (
                                        len(grayscale_cam1) * len(grayscale_cam2))).sum() ** 2
                            # print(mse)
                        break
                break
            ranked_layers.append((mse, layers[layer_no]))
            ranked_layers.sort(reverse=True)
            layer_no += 1
        print(ranked_layers)
        print()


traini_loss_c1 = []
valida_loss_c1 = []
traini_loss_c2 = []
valida_loss_c2 = []
traini_loss_c3 = []
valida_loss_c3 = []


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

                # Add the FedProx regularization term
                for param1, param2 in zip(model.parameters(), server.parameters()):
                    loss += (mu / 2) * torch.norm((param1.data - param2.data), p=2)

                # Backward Prop
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            val_loss = get_val_loss(model, criterion, val_loader, server)

            if i==0:
                valida_loss_c1.append(val_loss)
                traini_loss_c1.append(running_loss / len(c_train_loaders[i]))

            if i==1:
                valida_loss_c2.append(val_loss)
                traini_loss_c2.append(running_loss / len(c_train_loaders[i]))

            if i==2:
                valida_loss_c3.append(val_loss)
                traini_loss_c3.append(running_loss / len(c_train_loaders[i]))


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
cent_optimizier = optim.Adagrad(cent_model.parameters(), lr=clr)
cent_model = update_centralized_model(cent_model, models, num_clients)
num_communications = 9

print("---Centralized Model---")
acc, f1_sc, auroc = get_accuracy(cent_model, test_loader)
print("Communication", 1, "| Test Accuracy =", acc, "| F1-Score =", f1_sc, "| Auroc Score =", auroc)

for i in range(num_communications):
    models = update_client_models(cent_model, models, num_clients)
    train_clients(num_clients, cent_model, models, optimizers, criterions)
    cent_model = update_centralized_model(cent_model, models, num_clients)
    cent_optimizier.step()
    acc, f1_sc, auroc = get_accuracy(cent_model, test_loader)
    print("Communication", i + 2, "| Test Accuracy =", acc, "| F1-Score =", f1_sc, "| Auroc Score =", auroc)

models = update_client_models(cent_model, models, num_clients)
get_ranked_layers(models, num_clients, c_train_loaders)

print("---Client Models---")
for i in range(num_clients):
    model_name = "model" + str(i)
    acc, f1_sc, auroc = get_accuracy(models[model_name], test_loader)
    print(f"Test Accuracy for Client-{i + 1} is:-", acc)
    print(f"F1-Score for Client-{i + 1} is:-", f1_sc)
    print(f"Auroc Score for Client-{i + 1} is:-", auroc)

print("Training Loss Client1:-", traini_loss_c1)
print("Validation Loss Client1:-", valida_loss_c1)

print("Training Loss Client2:-", traini_loss_c2)
print("Validation Loss Client2:-", valida_loss_c2)

print("Training Loss Client3:-", traini_loss_c3)
print("Validation Loss Client3:-", valida_loss_c3)
