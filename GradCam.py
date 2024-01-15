import numpy as np
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision import transforms
from MobileNetModel import MobileNetV2
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

dataset = datasets.ImageFolder(root='Overall-Dataset', transform=transform)

dataloader = DataLoader(dataset, shuffle=False, batch_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        mobilenet = MobileNetV2(2)
        mobilenet.load_state_dict(torch.load('Trained Weights/proxopt_corr_trained_weights.pt', map_location=device))

        self.model = mobilenet

        self.features_conv = self.model.model[:1]

        self.max_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.classifier = nn.Linear(32, 2)

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)

        h = x.register_hook(self.activations_hook)

        x = self.max_pool(x)

        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features_conv(x)


mobilenetv2 = MobileNet()
mobilenetv2.eval()

img, _ = next(iter(dataloader))

pred = mobilenetv2(img)

pred[:1, 1].backward()

gradients = mobilenetv2.get_activations_gradient()

pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

# print(len(pooled_gradients))

activations = mobilenetv2.get_activations(img).detach()

for i in range(32):
    activations[:, i, :, :] *= pooled_gradients[i]

heatmap = torch.mean(activations, dim=1).squeeze()

heatmap = np.maximum(heatmap, 0)

heatmap /= torch.max(heatmap)

plt.matshow(heatmap.squeeze())
plt.show()

# img = cv2.imread('')
# heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
# heatmap = np.uint8(255 * heatmap)
# heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# superimposed_img = heatmap * 0.4 + img
# cv2.imwrite('./map.jpg', superimposed_img)
