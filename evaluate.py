import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model import Model
from dataset import UCMercedDataset, make_splits
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os

# 1. setup — device, model, load checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
os.makedirs("checkpoints", exist_ok = True)

# 2. setup — test dataset and loader (same val_transform as training)

val_transform = transforms.Compose([
                    transforms.Resize((128,128)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5)) ])

train_samples, val_samples, test_samples, class_to_idx = make_splits("UCMerced_LandUse/Images")

test_dataset = UCMercedDataset(test_samples, transform=val_transform)

test_loader = DataLoader(test_dataset, 32, False, num_workers = 0)

# 3. collect predictions
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        pred = model(X)
        all_preds.extend(pred.argmax(1).cpu().numpy())
        all_labels.extend(y.numpy())

# 4. accuracy
accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# 5. confusion matrix
class_names = list(class_to_idx.keys())
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(16, 16))
disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
plt.title("Confusion Matrix — UC Merced Test Set")
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.show()





























