import torch
import torch.nn as nn
import torch.nn.functional as f

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # BLOCK 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)

        # BLOCK 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)

        # BLOCK 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2)

        # Activation
        self.relu = nn.ReLU()

        # POOLING
        self.MaxPool = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(4, 4))
        
        # Fully-Connected Layers 
        self.fc1 = nn.Linear(in_features=128 * 4 * 4, out_features=625)
        self.fc2 = nn.Linear(in_features=625, out_features=10)

        # Drop-out
        self.DropOut = nn.Dropout(p=0.2)

        

    def forward(self, x):
        # x lúc này chính là lô ảnh có kích thước [batch_size, 1, 28, 28]

        # BLOCK 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.MaxPool(x)
        x = self.DropOut(x)

        # BLOCK 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.MaxPool(x)
        x = self.DropOut(x)

        # BLOCK 3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.MaxPool(x)

        x = self.adaptive_pool(x)

        # FLATTEN
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.DropOut(x)
        x = self.fc2(x)

        return x

# --- Code kiểm tra nhanh hình dáng dữ liệu ---
if __name__ == "__main__":
    model = CNN()
    dummy_input = torch.randn(64, 1, 28, 28)
    dummy_input_large = torch.randn(64, 1, 224, 224)
    
    output = model(dummy_input_large)
    print(output.shape)

    