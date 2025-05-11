import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ========= Command line arguments =========
parser = argparse.ArgumentParser(description="Predict flower class from an image.")
parser.add_argument('--image', type=str, required=True, help="Path to the input image")
args = parser.parse_args()

# ========= Define preprocessing =========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ========= Load class names =========
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# ========= Load model =========
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 5)
model.load_state_dict(torch.load("flower_resnet18.pth", map_location='cpu'))
model.eval()

# ========= Load and preprocess image =========
image = Image.open(args.image).convert("RGB")
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# ========= Run inference =========
with torch.no_grad():
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_class = class_names[predicted.item()]

print(f"Predicted class: {predicted_class}")

