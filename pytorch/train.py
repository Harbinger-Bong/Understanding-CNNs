# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_utils import create_model
import os

# 0. Setup Device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load Data & Preprocessing
# Transforms: Convert to Tensor (0-1 float) and Normalize (Mean, Std)
transform = transforms.Compose([
    transforms.ToTensor(), # Converts [H, W, C] to [C, H, W] and scales to 0-1
])

# Download and load training data
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# 2. Create Model and move to Device
print("Building model...")
model = create_model().to(device)

# 3. Define Loss and Optimizer
criterion = nn.CrossEntropyLoss() # Combines LogSoftmax + NLLLoss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training Loop (The explicit manual loop)
print("Starting training...")
epochs = 5

for epoch in range(epochs):
    model.train() # Set model to training mode (enables Dropout)
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # A. Zero gradients (reset from previous step)
        optimizer.zero_grad()
        
        # B. Forward Pass
        outputs = model(data)
        
        # C. Calculate Loss
        loss = criterion(outputs, target)
        
        # D. Backward Pass (Calculate gradients)
        loss.backward()
        
        # E. Update Weights
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# 5. Save the Model
save_path = 'mnist_cnn.pth'
# In PyTorch, we usually save the 'state_dict' (the weights dictionary)
torch.save(model.state_dict(), save_path)
print(f"Model weights saved successfully to {save_path}")