import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

from model import CNN

# ==========================================
# INITIALIZATION
# ==========================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = CNN().to(device)
model.load_state_dict(torch.load("mnist_paper_cnn.pth", map_location=device, weights_only=True))
model.eval()

# ==========================================
# INFERENCE
# ==========================================
def predict_my_image(image_path):
    print(f"[*] Loading image: {image_path}...")
    
    try:
        # Convert to grayscale and invert colors to match MNIST (white text, black background)
        img = Image.open(image_path).convert('L')
        img = ImageOps.invert(img)
        
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Apply transform and add batch dimension
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted_class = torch.max(output, 1)
            
            probabilities = F.softmax(output, dim=1)[0]
            confidence = probabilities[predicted_class.item()] * 100
            
        print("-" * 50)
        print(f"[*] PREDICTION : {predicted_class.item()}")
        print(f"[*] CONFIDENCE : {confidence:.2f}%")
        print("-" * 50)
        
        plt.imshow(img_tensor.cpu().numpy().squeeze(), cmap='gray')
        plt.title(f"Predicted: {predicted_class.item()} | Confidence: {confidence:.1f}%")
        plt.axis('off')
        plt.show()
        
    except FileNotFoundError:
        print(f"[ERROR] File '{image_path}' not found.")

if __name__ == '__main__':
    predict_my_image("your_picture.png")