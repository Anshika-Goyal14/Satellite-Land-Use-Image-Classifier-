import torch 
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
from torchvision import transforms
from model import Model
from dataset import UCMercedDataset, make_splits 
from collections import defaultdict 
from PIL import Image 

def get_gradcam(model, image_tensor, class_idx):
    store = {}


    def forward_hook(module, input, output):
        store['activations'] = output.detach()


    def backward_hook(module, grad_input, grad_output):
        store['gradients'] = grad_output[0].detach()

    h1 = model.block4[0].register_forward_hook(forward_hook)
    h2 = model.block4[0].register_full_backward_hook(backward_hook)

    model.eval()
    output = model(image_tensor)

    model.zero_grad()
    output[0, class_idx].backward()

    gradients = store['gradients']
    activations = store['activations']

    weights = gradients.mean(dim=(2,3), keepdim= True)
    heatmap = (weights * activations).sum(dim=1).squeeze()
    heatmap = torch.relu(heatmap)

    heatmap = heatmap/(heatmap.max()+ 1e-8)

    h1.remove()
    h2.remove()
    return heatmap.cpu().numpy()



def show_gradcam(image_tensor, heatmap, class_name, save = True):
    img = image_tensor.squeeze().permute(1,2,0).cpu().numpy()

    img = (img*0.5) + 0.5
    img = np.clip(img, 0,1)
    heatmap_resized = cv.resize(heatmap,(128,128))

    heatmap_colored = plt.cm.jet(heatmap_resized)[:,:,:3]
    overlay = 0.6*img + 0.4*heatmap_colored
    fig,axes = plt.subplots(1,3, figsize = (12,4))

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title("Grad-CAM heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay — {class_name}")
    axes[2].axis("off")

    plt.tight_layout()

    if save:
        os.makedirs("outputs/gradcam", exist_ok=True)
        plt.savefig(f"outputs/gradcam/{class_name}.png", dpi=150)

    plt.show()
    plt.close()

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model().to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])


    _, _, test_samples, class_to_idx = make_splits("UCMerced_LandUse/Images")
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    samples_by_class = defaultdict(list)
    for filepath, label in test_samples:
        samples_by_class[label].append(filepath)

    for class_idx in sorted (samples_by_class.keys()):
        filepath = samples_by_class[class_idx][0]
        class_name = idx_to_class[class_idx]
        img = Image.open(filepath).convert("RGB")
        image_tensor = transform(img).unsqueeze(0).to(device)

        heatmap = get_gradcam(model, image_tensor, class_idx)

        print(f"Grad-CAM: {class_name}")
        show_gradcam(image_tensor, heatmap, class_name)

    











