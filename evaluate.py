import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model import Model
from dataset import UCMercedDataset, make_splits
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

# 1. setup — device, model, load checkpoint
# 2. setup — test dataset and loader (same val_transform as training)
# 3. collect all predictions and true labels over the test set
#    (model.eval() + torch.no_grad(), loop over test_loader)
# 4. compute accuracy
# 5. confusion matrix — use sklearn, plot with class names, save as PNG
