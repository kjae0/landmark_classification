# Generate augmentation dataset
# Rotation, Cropping, Erasing


import torch.utils.data as data
import os
from PIL import Image
import pandas as pd
import numpy as np
from torchvision.transforms import functional, transforms
import custom_dataset

data_dir = ""

dataset = custom_dataset.Dataset(os.path.join(data_dir, 'train'), 
                                os.path.join(data_dir, 'train.csv'))

def save_images(images, type, file_names):
    save_dir = "C://Users/rlawo/Desktop/Dacon/landmark_classification/dataset/train_augmented/"+type

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(len(file_names))
    unit = len(file_names)//10
    cnt = 0
    for file_name, image in zip(file_names, images):
        image.save(os.path.join(save_dir, file_name), "png")
        cnt += 1
        if cnt%unit==0:
            print(len(file_names)//cnt)
        
        
crop = []
labels = []
image_shape = [540, 960]


for x, y in dataset:
    for i in range(4):
        height = np.random.randint(300, 540)
        width = np.random.randint(600, 960)
        pos = [np.random.randint(0, image_shape[0]-height), np.random.randint(0, image_shape[1]-width)]
        crop.append(functional.crop(x, pos[0], pos[1], height, width))
        labels.append(y)
        
df = pd.DataFrame(labels, columns=['label'])
file_names = [f'{i}.PNG' for i in range(1, len(crop)+1)]
df['file_name'] = [f'{i}.PNG' for i in range(1, len(crop)+1)]
df.to_csv("C://Users/rlawo/Desktop/Dacon/landmark_classification/dataset/train_augmented_random_crop.csv")
save_images(crop, 'random_crop', file_names)


erasing = []
labels = []
image_shape = [540, 960]
erasing_size = 200

for x, y in dataset:
    x = transforms.ToTensor()(x)
    for i in range(4):
        pos = [np.random.randint(0, 340), np.random.randint(0, 760)]
        image = functional.erase(x, pos[0], pos[1], erasing_size, erasing_size, v=0)
        image = transforms.ToPILImage()(image)
        erasing.append(image)
        labels.append(y)
        

df = pd.DataFrame(labels, columns=['label'])
file_names = [f'{i}.PNG' for i in range(1, len(erasing)+1)]
df['file_name'] = [f'{i}.PNG' for i in range(1, len(erasing)+1)]
df.to_csv("C://Users/rlawo/Desktop/Dacon/landmark_classification/dataset/train_augmented_erasing.csv")
save_images(erasing, 'erasing', file_names)