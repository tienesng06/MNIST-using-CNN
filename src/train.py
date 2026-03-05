import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os 

from model import CNN
from data_pipeline import train_dataset, test_dataset

# ==========================================
# DEVICE SETUP AND DATA LOADER
# ==========================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[*] Training on: {device}")

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# ==========================================
# TRAINING
# ==========================================
def train_model(model, criterion, optimizer, num_epochs=10):
    print("\n" + "="*50)
    print("[*] TRAINING PROCESS :")
    print("="*50)
    
    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}")
                running_loss = 0.0

    print("[*] TRAINING COMPLETE !")
    
    # SAVE MODEL
    torch.save(model.state_dict(), "mnist_paper_cnn.pth")
    print("[*] Đã lưu trọng số vào file 'mnist_paper_cnn.pth'")

# ==========================================
# EVALUATION
# ==========================================
def evaluate_model(model, weight_path="mnist_paper_cnn.pth"):
    print("\n" + "-"*50)
    print("[*] TESTING")
    
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        print(f"[*] model saved : {weight_path}")
    else:
        print(f"[WARNING] FILE NOT EXIST {weight_path}. TESTING WITH UNTRAINED MODEL!")

    model.eval() 
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = 100 * correct_predictions / total_samples
    print(f"[*] ACCURACY : {accuracy:.2f}%")
    print("-" * 50)


if __name__ == "__main__":
    my_model = CNN().to(device)
    my_criterion = nn.CrossEntropyLoss()
    my_optimizer = optim.RMSprop(my_model.parameters(), lr=0.001, alpha=0.9)
    
    # train_model(my_model, my_criterion, my_optimizer, num_epochs=10)

    evaluate_model(my_model)