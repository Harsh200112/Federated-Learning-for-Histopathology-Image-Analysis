import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from MobileNetModel import MobileNetV2
from torchvision.transforms import transforms
from PIL import Image
import matplotlib.pyplot as plt

model = MobileNetV2(2)
model.load_state_dict(torch.load('Trained Weights/proxopt_corr_trained_weights.pt'))

print(model)

target_layers = [model.avgpool]

image_path = "Bracs/Benign/BRACS_264_N_1.jpg"
input_image = Image.open(image_path)

# Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image).unsqueeze(0)

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers)

targets = [ClassifierOutputTarget(0)]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

input_image_np = input_tensor.squeeze().numpy().transpose(1, 2, 0)
input_image_np = (input_image_np - input_image_np.min()) / (input_image_np.max() - input_image_np.min())

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

for name, named_parameters in model.named_parameters():
    print(name)

org_image = transform(input_image)
org_image = org_image.numpy().transpose(1, 2, 0)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
print(len(grayscale_cam))

print("MSE =", 0.5 * ((grayscale_cam - org_image)/(len(grayscale_cam) * len(grayscale_cam))).sum()**2)

visualization = show_cam_on_image(input_image_np, grayscale_cam, use_rgb=True)

plt.title('2nd Inverted Residual Block')
plt.imshow(visualization)
plt.axis('off')
plt.show()
