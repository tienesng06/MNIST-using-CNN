import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random


from model import CNN
from data_pipeline import test_dataset

# ==========================================
# 1. SETUP
# ==========================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = CNN().to(device)

model.load_state_dict(torch.load("mnist_paper_cnn.pth", map_location=device, weights_only=True))

model.eval()


# ==========================================
# 2. PREPARE AND PREDICT
# ==========================================
def predict_random_image():
    idx = random.randint(0, len(test_dataset) - 1)
    img_tensor, actual_label = test_dataset[idx]
    
    img_batch = img_tensor.unsqueeze(0).to(device)
    
    
    with torch.no_grad():
        output = model(img_batch)
        
        _, predicted_class = torch.max(output, 1)
        
        # SOFTMAX
        probabilities = F.softmax(output, dim=1)[0]
        confidence = probabilities[predicted_class.item()] * 100
        
    print("-" * 50)
    print(f"[*] Ground Truth: Num {actual_label}")
    print(f"[*] Predict           : Num {predicted_class.item()}")
    print(f"[*] Confidence   : {confidence:.2f}%")
    print("-" * 50)
    
    plt.imshow(img_tensor.numpy().squeeze(), cmap='gray')
    plt.title(f"PREDICT: {predicted_class.item()} | RESULT: {actual_label} \n(confidence: {confidence:.1f}%)")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    predict_random_image()