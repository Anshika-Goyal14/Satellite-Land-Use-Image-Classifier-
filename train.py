import torch
import torch.nn as nn
from PIL import Image
import torchvision
from torchvision import transforms
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataset import UCMercedDataset, make_splits
from model import Model
import os

train_transform = transforms.Compose([
                            transforms.Resize((128,128)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean= (0.5,0.5,0.5), std = (0.5,0.5,0.5))
                                ])

val_transform = transforms.Compose([
                    transforms.Resize((128,128)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))
                       ])

train_samples, val_samples, test_samples, class_to_idx = make_splits("UCMerced_LandUse/Images")

train_dataset = UCMercedDataset(train_samples, transform = train_transform)
val_dataset = UCMercedDataset(val_samples, transform = val_transform)
test_dataset = UCMercedDataset(test_samples, transform=val_transform)

train_loader = DataLoader(train_dataset, 32, True, num_workers = 0)
val_loader = DataLoader(val_dataset, 32, False, num_workers = 0)
test_loader = DataLoader(test_dataset, 32, False, num_workers = 0)

print(len(train_dataset), len(val_dataset), len(test_dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
print(f"Using device: {device}")

learning_rate = 0.0001
batch_size = 64
epochs  = 80

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience = 5, factor = 0.5, verbose = True)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    running_loss = 0
    correct = 0

    for X, y in dataloader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        pred = model(X)
        loss = loss_fn(pred,y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (pred.argmax(1) == y).sum().item()
    average = running_loss / len(dataloader)
    train_accuracy = correct /len(dataloader.dataset)
    return average, train_accuracy 

def test_loop (dataloader, model, loss_fn):
    model.eval()

    running_loss =0 
    

    size = len(dataloader.dataset)
    num_bataches = len(dataloader)
    test_loss, correct = 0,0

    with torch.no_grad():
        for X,y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)

            loss = loss_fn(pred, y)
            running_loss+= loss.item()
            correct += (pred.argmax(1)== y).sum().item()

    avg_val_loss = running_loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)

    return avg_val_loss, accuracy 

train_losses = []
val_losses = []

train_losses_lr_01 = train_losses.copy() 
train_losses_lr_001 = train_losses.copy()
train_losses_lr_0001 = train_losses.copy() 

os.makedirs("checkpoints", exist_ok = True)

best_val_acc = 0.0

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}")

    train_loss, train_acc = train_loop(train_loader, model, loss_fn, optimizer)
    val_loss, accuracy = test_loop(val_loader, model, loss_fn)
    train_losses. append(train_loss)
    val_losses.append(val_loss)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Accuracy   : {accuracy*100:.2f}%")

    scheduler.step(val_loss)

    if accuracy > best_val_acc:
        best_val_acc = accuracy
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
        print(f"  Saved best model (val acc: {accuracy*100:.2f}%)")

    if torch.cuda.is_available():
        print(f"  GPU memory: {torch.cuda.memory_allocated()/1024**2:.0f}MB")


from utils import save_losses, plot_losses

save_losses(train_losses, val_losses)
plot_losses()











        

