import json
import os
import matplotlib.pyplot as plt 

def save_losses (train_losses, val_losses , path = "outputs/losses.json"):
    os.makedirs("outputs", exist_ok = True)
    with open(path, "w") as f:         
        json.dump({                    
            "train": train_losses,
            "val": val_losses
        }, f)

def plot_losses(path="outputs/losses.json"):
    with open(path, "r") as f:         
        data = json.load(f)            

    train_losses = data["train"]        
    val_losses = data["val"]

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")   
    plt.plot(val_losses,   label="Val Loss")     
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()                    
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/loss_curve.png", dpi=150)
    plt.show()