import torch.utils.data as data
import os
from PIL import Image
import pandas as pd
from torchvision.transforms import functional, transforms




class Dataset(data.Dataset):
    def __init__(self, data_dir, csv_dir, transform=None):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.csv_dir = csv_dir
        self.df = pd.read_csv(csv_dir)
        self.labels = list(self.df['label'])
        self.file_names = self.df['file_name']
        self.image_shape = [540, 960]
        self.images = []
                
        for file_name in self.file_names:
            image_dir = os.path.join(self.data_dir, file_name)
            image = Image.open(image_dir).convert('RGB')
            if self.transform:
                image = self.transform(image)
            self.images.append(image)
            
    def categorized_images(self, category):
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

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.images)
    

class TestDataset(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(TestDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.file_names = self.df['file_name']
        self.images = []
                
        for file_name in self.file_names:
            image_dir = os.path.join(self.data_dir, file_name)
            image = Image.open(image_dir).convert('RGB')
            if self.transform:
                image = self.transform(image)
            self.images.append(image)

    def __getitem__(self, idx):
        return self.images[idx]
    
    def __len__(self):
        return len(self.images)
    