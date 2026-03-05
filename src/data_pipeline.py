import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# ---------------------------------------------------
# TRANSFORM
# ---------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081))
])


# ---------------------------------------------------
# DATASET
# ---------------------------------------------------

# Train set
train_dataset = datasets.MNIST(
    root='./Dataset/',
    train=True,
    download=True,
    transform=transform
)

# Test set
test_dataset = datasets.MNIST(
    root='./Dataset/',
    train=False,
    download=True,
    transform=transform
)

if __name__ == "__main__":
        
    # ---------------------------------------------------
    # DATA LOADER
    # ---------------------------------------------------
    batch_size = 128
    train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size, shuffle=True)


    print("-" * 30)
    print(f"Số lượng ảnh Train: {len(train_dataset)}")
    print(f"Số lượng ảnh Test: {len(test_dataset)}")
    print(f"Số lô (batch) trong tập Train: {len(train_loader)}")
    print("-" * 30)

    # ---------------------------------------------------
    # VISUALIZE
    # ---------------------------------------------------

    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    print(f"Kích thước 1 lô ảnh: {images.shape}") 

    print(f"Kích thước nhãn (labels): {labels.shape}")

    fig = plt.figure(figsize=(10, 4))
    for i in range(6):
        ax = fig.add_subplot(1, 6, i+1, xticks=[], yticks=[])
        ax.imshow(images[i].numpy().squeeze(), cmap='gray')
        ax.set_title(f"Label: {labels[i].item()}")

    plt.show()