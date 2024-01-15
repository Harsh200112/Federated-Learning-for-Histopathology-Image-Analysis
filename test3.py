import torch.nn as nn
import torch
from torchvision.models import mobilenetv2

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

# Example usage:
cent_model = MobileNetV2(2)

print("---Model Description---")
print(cent_model)

target_state_dict = cent_model.state_dict()
print()

print("---LAYERS---")
for key in target_state_dict:
    if target_state_dict[key].data.dtype == torch.float32:
        print(key)
        target_state_dict[key].data.fill_(0.)
        # for i in range(num_clients):
        #     model_name = "model" + str(i)
        #     model = models[model_name]
        #     state_dict = model.state_dict()
        #     target_state_dict[key].data += state_dict[key].data.clone()/num_clients

cent_model.load_state_dict(target_state_dict)
