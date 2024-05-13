import torch
import torch.nn as nn
from model import CNN
from torch.nn.functional import relu
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from dataloader import transform


class GradGAM:
    def __init__(self, model, feature_layer):
        self.model = model
        self.feature_layer = feature_layer
        self.gradients = None
        self.activations = None
        self.model.eval()
        self.hook_layers()
    
    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        self.feature_layer.register_backward_hook(hook_function)

        def forward_hook(module, input, output):
            self.feature_maps = output
        self.feature_layer.register_forward_hook(forward_hook)

    def generate_cam(self, input_image, target_class):
        model_output = self.model(input_image)
        self.model.zero_grad()
        class_loss = model_output[0, target_class]
        class_loss.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.feature_maps.shape[1]):
            self.feature_maps[:, i, :, :] *= pooled_gradients[i]
        
        cam = torch.mean(self.feature_maps, dim=1)[0]
        cam = relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes=6).to(device)
model.load_state_dict(torch.load('model2.pth', map_location=device))
model.to(device)
model.eval()
image_path = 'data/imgs/16175.jpg'
img = Image.open(image_path).convert('RGB')
input_image = transform(img).unsqueeze(0).to(device)
grad_cam = GradGAM(model, model.conv3)
cam = grad_cam.generate_cam(input_image, 5)
plt.imshow(cam.cpu().detach().numpy(),cmap='hot')
plt.show()