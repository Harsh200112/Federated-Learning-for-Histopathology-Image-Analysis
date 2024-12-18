# Importing Libraries
import torch
import numpy as np
import torch.nn as nn
import torchmetrics
from tqdm import tqdm
from functools import reduce
import torch.optim as optim
from pytorch_grad_cam import GradCAM
from MobileNetModel import MobileNetV2
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from DataLoaders import c_train_loaders, c_val_loaders, test_loader

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
k = 10

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
def get_val_loss(model, criterion, data, server, ranked_layers):
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for x, y in data:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            val_loss += criterion(scores, y)

            for (name1, param1), (name2, param2) in zip(model.named_parameters(), server.named_parameters()):
                if name1[:14] in ranked_layers or name1[:2] == 'fc':
                    val_loss += (mu / 2) * torch.norm((param1.data - param2.data), p=2)

    return val_loss.item() / len(data)


# Setting the Main Model Parameters
def update_centralized_model(cent_model, models, num_clients, layers):
    target_state_dict = cent_model.state_dict()
    temp_state_dict = cent_model.state_dict()
    # print(layers)

    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            target_state_dict[key].data.fill_(0.)
            for i in range(num_clients):
                model_name = "model" + str(i)
                model = models[model_name]
                state_dict = model.state_dict()
                target_state_dict[key].data += state_dict[key].data.clone() / num_clients
                if key[:14] in layers or key[:2] == 'fc':
                    if i == 0:
                        target_state_dict[key].grad = (state_dict[key].data.clone() - temp_state_dict[
                            key].data.clone()) / num_clients
                    else:
                        target_state_dict[key].grad += (state_dict[key].data.clone() - temp_state_dict[
                            key].data.clone()) / num_clients

    cent_model.load_state_dict(target_state_dict)
    return cent_model


# Finding the Rank of Layers important in predicting the final outcome
def get_ranked_layers(clients, num_clients, c_train_loaders, top_ranks):
    model_ranked_layers = {}
    layers = []
    for key in clients["model0"].state_dict():
        if key[:14] not in layers:
            layers.append(key[:14])
    # print(layers)
    # print(len(layers))
    for no in tqdm(range(num_clients)):
        model_name = "model" + str(no)
        model = clients[model_name]
        model.eval()
        target_layers = [[model.model[0].conv[0]], [model.model[0].conv[1]],
                         [model.model[1].conv[0]], [model.model[1].conv[1]],
                         [model.model[1].conv[3]], [model.model[1].conv[4]], [model.model[2].conv[0]],
                         [model.model[2].conv[1]], [model.model[2].conv[3]],
                         [model.model[2].conv[4]], [model.model[2].conv[6]],
                         [model.model[2].conv[7]],
                         [model.model[3].conv[0]], [model.model[3].conv[1]]]
        ranked_layers = []
        # print(len(target_layers))
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

                            mse += 1 / batch_size * ((grayscale_cam1 - grayscale_cam2) / (
                                    len(grayscale_cam1) * len(grayscale_cam2))).sum() ** 2
                            # print(mse)
                        break
                break
            ranked_layers.append((mse, layers[layer_no]))
            layer_no += 1
        # print(ranked_layers)
        # print()
        ranked_layers.sort(reverse=True)
        model_ranked_layers[model_name] = [t[1] for t in ranked_layers]

    list = []
    for i in range(num_clients):
        model_name = "model" + str(i)
        list.append(model_ranked_layers[model_name][:top_ranks])

    layers = reduce(np.intersect1d, list)

    return layers


overall_train_loss = []
overall_val_loss = []
traini_loss_c1 = []
valida_loss_c1 = []
traini_loss_c2 = []
valida_loss_c2 = []
traini_loss_c3 = []
valida_loss_c3 = []


# Training Clients
def train_clients(num_clients, server, models, optimizers, criterions, ranked_layers):
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
                for (name1, param1), (name2, param2) in zip(model.named_parameters(), server.named_parameters()):
                    if name1[:14] in ranked_layers or name1[:2] == 'fc':
                        # print("reached here!!", name1[:2])
                        loss += (mu / 2) * torch.norm((param1.data - param2.data), p=2)

                # Backward Prop
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            val_loss = get_val_loss(model, criterion, c_val_loaders[i], server, ranked_layers)
            overall_train_loss.append(running_loss/len(c_train_loaders[i]))
            overall_val_loss.append(val_loss)

            if i == 0:
                valida_loss_c1.append(val_loss)
                traini_loss_c1.append(running_loss / len(c_train_loaders[i]))

            if i == 1:
                valida_loss_c2.append(val_loss)
                traini_loss_c2.append(running_loss / len(c_train_loaders[i]))

            if i == 2:
                valida_loss_c3.append(val_loss)
                traini_loss_c3.append(running_loss / len(c_train_loaders[i]))

    return models


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
num_communications = 10

print("---Centralized Model---")
ranked_client_layers = []

for i in range(num_communications):
    cent_model.train()
    models = update_client_models(cent_model, models, num_clients)
    models = train_clients(num_clients, cent_model, models, optimizers, criterions, ranked_client_layers)
    ranked_client_layers = get_ranked_layers(models, num_clients, c_train_loaders, k)
    print(ranked_client_layers)
    cent_model = update_centralized_model(cent_model, models, num_clients, ranked_client_layers)
    cent_optimizier.step()
    acc, f1_sc, auroc = get_accuracy(cent_model, test_loader)
    print("Communication", i + 1, "| Test Accuracy =", acc, "| F1-Score =", f1_sc, "| Auroc Score =", auroc)

models = update_client_models(cent_model, models, num_clients)

print("\n")
print("---Client Models---")
for i in range(num_clients):
    model_name = "model" + str(i)
    acc, f1_sc, auroc = get_accuracy(models[model_name], test_loader)
    print(f"Test Accuracy for Client-{i + 1} is:-", acc)
    print(f"F1-Score for Client-{i + 1} is:-", f1_sc)
    print(f"Auroc Score for Client-{i + 1} is:-", auroc)

print("\n")
print("Overall Training Loss:-", overall_train_loss)
print("Overall Validation Loss:-", overall_val_loss, "\n")

print("Training Loss Client1:-", traini_loss_c1)
print("Validation Loss Client1:-", valida_loss_c1, "\n")

print("Training Loss Client2:-", traini_loss_c2)
print("Validation Loss Client2:-", valida_loss_c2, "\n")

print("Training Loss Client3:-", traini_loss_c3)
print("Validation Loss Client3:-", valida_loss_c3)
