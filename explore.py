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
import os 
from collections import Counter 


#Exploring the Dataset -----


path = "UCMerced_LandUse/Images"
for root, dirs, files in os.walk(path):
    print(f"Folder : {root}")
    print(f"Subfolders : {dirs}")
    print(f"Number of files : {len(files)}")
    print("-"*40)




class_counts = {}


for root, dirs, files in os.walk(path):


    if root == path:
        continue


    class_name = os.path.basename(root)


    class_counts[class_name] = len(files)


print(class_counts)




#Opening 5 images per class just to see familiarity ---
'''for class_name in sorted(os.listdir(path)):


    class_path = os.path.join(path, class_name)


    if not os.path.isdir(class_path):
        continue


    images = sorted(os.listdir(class_path))[:5]


    plt.figure(figsize=(15,3))
    plt.suptitle(class_name, fontsize=16)


    for i, image_name in enumerate(images):


        image_path = os.path.join(class_path, image_name)


        img = Image.open(image_path)


        plt.subplot(1,5,i+1)
        plt.imshow(img)
        plt.axis("off")


    plt.tight_layout()
    plt.show()'''






#checking size of all the images 
sizes = []


for class_name in os.listdir(path):


    class_path = os.path.join(path, class_name)


    if not os.path.isdir(class_path):
        continue


    for image_name in os.listdir(class_path):


        image_path = os.path.join(class_path, image_name)


        img = Image.open(image_path)


        sizes.append(img.size)


print(set(sizes))


#to get the count of each image size 
size_counts = Counter(sizes) 


for size, count in sorted(size_counts.items()):
    print(f"{size}: {count}") #2056 images show the size of 256,256. the rest 44 images differ by a few pixels. 




#outlier images -
for class_name in os.listdir(path):
    class_path = os.path.join(path,class_name)
    if not os.path.isdir(class_path):
        continue


    for image_name in os.listdir(class_path):


        image_path = os.path.join(class_path, image_name)
        img = Image.open(image_path)


        if img.size!= (256,256):
            print(image_path, img.size) #can resize these images 

