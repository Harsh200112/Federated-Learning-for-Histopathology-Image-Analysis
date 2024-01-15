import torch
import torch.nn as nn
from MobileNetModel import MobileNetV2
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2


# Define the GradCAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        # Register hook to capture gradients
        self.hook = self.register_hooks()

    def register_hooks(self):
        def hook_fn(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_layer = self.model._modules.get(self.target_layer)
        hook = target_layer.register_forward_hook(hook_fn)
        return hook

    def remove_hooks(self):
        self.hook.remove()

    def forward(self, x):
        return self.model(x)

    def backward(self, target_class):
        self.model.zero_grad()
        one_hot_output = torch.zeros_like(self.gradients)
        one_hot_output[:, target_class] = 1
        self.gradients.backward(gradient=one_hot_output, retain_graph=True)

    def get_activations_and_gradients(self, x):
        output = self.forward(x)
        target_class = torch.argmax(output)
        self.backward(target_class)
        return output, self.gradients

    def generate_cam(self, x, size):
        output, gradients = self.get_activations_and_gradients(x)
        activations = output[0, :]

        # Check if the gradients tensor has additional dimensions
        if len(gradients.shape) > 2:
            # Squeeze additional dimensions
            gradients = gradients.squeeze(0)

        # Take the mean across channels (axis=0)
        weights = torch.mean(gradients, axis=(1, 2), keepdim=True)

        # Sum the weighted activations across channels
        cam = torch.sum(weights * activations, axis=0)
        cam = torch.relu(cam)

        cam = cam.detach().numpy()
        cam = cv2.resize(cam, size)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam


# Function to apply Grad-CAM to every layer of a model
def apply_gradcam_to_layers(model, image_path, size=(224, 224)):
    for name, layer in model.named_children():
        print(f"Applying Grad-CAM to layer: {name}")

        # Initialize GradCAM for the current layer
        gradcam = GradCAM(model, name)

        # Load and preprocess the image
        preprocess = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path)
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        # Generate CAM
        cam = gradcam.generate_cam(input_batch, size)

        # Superimpose CAM on the original image
        img_array = np.array(image)
        cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam = np.float32(cam) / 255.0
        # cam = cam + np.float32(img_array) / 255.0
        cam = cam / np.max(cam)

        # Display the images
        # cv2.imshow(f'Original Image - {name}', img_array)
        cv2.imshow(f'Grad-CAM - {name}', cam)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage:
# Replace 'your_model' with an instance of the MobileNetV2 model
# Replace 'your_image.jpg' with the path to the image you want to analyze
# Make sure the image size matches the input size of your model
# apply_gradcam_to_layers(your_model, 'your_image.jpg')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = MobileNetV2(2)
model.load_state_dict(torch.load('Trained Weights/proxopt_corr_trained_weights.pt',map_location=device))

apply_gradcam_to_layers(model, 'Overall-Dataset/Benign/BRACS_291_PB_3.jpg')