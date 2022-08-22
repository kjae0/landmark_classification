import torch.utils.data as data
import os
from PIL import Image
import pandas as pd
import augmentation
from torchvision.transforms import functional



class Dataset(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.file_names = os.listdir(data_dir)
        self.transform = transform
        self.images = []
        for file_name in self.file_names:
            image_dir = os.path.join(self.data_dir, file_name)
            image = Image.open(image_dir).convert('RGB')
            if self.transform:
                image = self.transform(image)
            self.images.append(image)
            
    def categorized_images(self, category, csv_dir):
        self.csv_dir = csv_dir
        self.label_images = []
        
        labels_csv = pd.read_csv(self.csv_dir)
        labels = labels_csv['label']
        labels_indices = labels==category
        labels_file_names = labels_csv['file_name'][labels_indices]
        
        for label_file_name in labels_file_names:
            image_dir = os.path.join(self.data_dir, label_file_name)
            image = Image.open(image_dir).convert('RGB')
            if self.transform:
                image = self.transform(image)
            self.label_images.append(image)

        return self.label_images

    def rotation(self, unit, maximum=360):
        rotated_images = []
        for image in self.images:
            for degree in range(1, int(maximum//unit)):
                rotated_images.append(functional.rotate(image, unit*degree))
        print(f'{len(rotated_images)} images generated.')
        return rotated_images
        

    def cropping(self, unit, height, width):
        cropped_images = []
        for image in self.images:
            for col in range(int((image.shape[1]-width)//unit)+1):
                for row in range(int((image.shape[0]-height)//unit)+1):
                    cropped_images.append(functional.crop(image, col, row, height, width))
        print(f'{len(cropped_images)} images generated.')
        return cropped_images

    def brighteness(self, unit, maximum=2, minimum=0):
        modified_images = []
        for image in self.images:
            for degree in range(int(minimum//unit), int(maximum//unit)+1):
                modified_images.append(functional.adjust_brightness(image, unit*degree))
        print(f'{len(modified_images)} images generated.')
        return modified_images

    def erasing(self, unit, height, width):
        erased_images = []
        for image in self.images:
            for col in range(int((image.shape[1]-width)//unit)+1):
                for row in range(int((image.shape[0]-height)//unit)+1):
                    erased_images.append(functional.erase(image, col, row, height, width))
        print(f'{len(erased_images)} images generated.')
        return erased_images

    def __getitem__(self, idx):
        return self.images[idx]
    
    def __len__(self):
        return len(self.images)
    
    