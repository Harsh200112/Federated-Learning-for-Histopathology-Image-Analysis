import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import datasets
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from MobileNetModel import MobileNetV2
from torchvision.transforms import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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

# print(classes)


def train_test_ds(data, test_split=0.3):
    train_idx, val_idx = train_test_split(list(range(len(data))), test_size=test_split)
    train_data = Subset(data, train_idx)
    test_data = Subset(data, val_idx)

    return train_data, test_data


train_data, x_data = train_test_ds(dataset)
test_data, val_data = train_test_ds(x_data, 1/3)

# DataLoaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)

# Model
model = MobileNetV2(2).to(device)
model.load_state_dict(torch.load('Trained Weights/proxopt_corr_trained_weights.pt', map_location=device))


print(model)

# print(model.state_dict().keys())
#
# layers = []
#
# for key in model.state_dict():
#     if key[:14] not in layers:
#         layers.append(key[:14])
#
# print(layers)
#
# print(type(model.model[0].conv[1]))
# print(target_layers)

# layers = ['Conv Block-1', 'Inverted Residual Block-1 1st Conv', 'Inverted Residual Block-1 2nd Conv', 'Inverted Residual Block-2 1st Conv', 'Inverted Residual Block-2 2nd Conv', 'Inverted Residual Block-2 3rd Conv', 'Conv Block-2', 'Avg Pool']
# target_layers = [[model.model[0].conv[0]], [model.model[1].conv[0]], [model.model[1].conv[3]], [model.model[2].conv[0]], [model.model[2].conv[3]],
#                  [model.model[2].conv[6]], [model.model[3].conv[0]], [model.avgpool]]
# ranked_layers = []
#
# print("----Ranked Layers For Client", 1,"----")
# layer_no = 0
# for layer in target_layers:
#     mse = 0
#     for x, y in val_loader:
#         x = x.to(device)
#         y = y.to(device)
#         for i in range(batch_size):
#             for l, k in val_loader:
#                 l = l.to(device)
#                 k = k.to(device)
#                 for j in range(batch_size):
#                     cam = GradCAM(model=model, target_layers=layer)
#                     target1 = [ClassifierOutputTarget(y[i])]
#                     target2 = [ClassifierOutputTarget(k[j])]
#
#                     grayscale_cam1 = cam(input_tensor=x[i].unsqueeze(0), targets=target1)
#                     grayscale_cam2 = cam(input_tensor=l[j].unsqueeze(0), targets=target2)
#
#                     grayscale_cam1 = grayscale_cam1[0, :]
#                     grayscale_cam2 = grayscale_cam2[0, :]
#
#                     mse += 1/batch_size * ((grayscale_cam1 - grayscale_cam2) / (
#                                 len(grayscale_cam1) * len(grayscale_cam2))).sum() ** 2
#                     # print(mse)
#                 break
#         break
#     ranked_layers.append((mse, layers[layer_no]))
#     ranked_layers.sort(reverse=True)
#
#     layer_no += 1
# print(ranked_layers)
# print()
#
#
# print('Conv Block-1' in ranked_layers)