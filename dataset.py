import os
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class UCMercedDataset(Dataset):
    def __init__(self, samples, transform= None):
      self.samples = samples
      self.transform = transform 

      pass

    def __len__(self):
       return len(self.samples)

    def __getitem__(self,idx):
        filepath, label = self.samples[idx]
        image = Image.open(filepath).convert("RGB")

        if self.transform :
           image = self.transform(image)

        return image, label


def make_splits(root_path, val_size = 0.15, test_size = 0.15, random_state = 42):

       class_names = sorted([ d for d in os.listdir(root_path)
                            if os.path.isdir(os.path.join(root_path,d))])

       class_to_idx = {name : idx for idx, name in enumerate (class_names)}

       samples = []
       for class_name in class_names:
          class_path  = os.path.join(root_path, class_name)
          for fname in os.listdir(class_path):
             if fname.lower().endswith((".jpg",".png", ".tif")):
                full_path = os.path.join(class_path, fname)
                samples.append((full_path, class_to_idx[class_name]))

       labels = [s[1] for s in samples]

       train_val_samples, test_samples, train_val_labels, _ = train_test_split(samples, labels, test_size = test_size, stratify = labels, random_state = random_state)

       relative_val_size = val_size / (1-test_size)

       train_samples, val_samples, _, _ = train_test_split(train_val_samples, train_val_labels, test_size = relative_val_size, stratify = train_val_labels, random_state=random_state)

       return train_samples, val_samples, test_samples, class_to_idx


    
    
    
    
    

































   