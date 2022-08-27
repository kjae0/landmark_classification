from torchvision import models
from torchvision.models import resnet50 as ResNet_50
from torchvision.models import inception_v3 as Inception_v3
from torchvision.models import efficientnet_b3 as efficientNet_b3
from torchvision.models import efficientnet_b4 as efficientNet_b4
import torch
import torch.nn as nn

class ResNet50(nn.Module):
    def __init__(self, num_labels=10, pretrained=False):
        super(ResNet50, self).__init__()
        self.num_labels = num_labels
        self.pretrained = pretrained
        
        self.model = ResNet_50(pretrained=self.pretrained)
        self.fc = nn.Sequential(nn.Linear(2048, 512),
                               nn.Linear(512, self.num_labels))
        self.model.fc = self.fc
        
    def forward(self, x):
        return self.model(x)
    

class EfficientNetB3(nn.Module):
    def __init__(self, num_labels=10, pretrained=False):
        super(EfficientNetB3, self).__init__()
        self.num_labels = 10
        self.pretrained = pretrained
        
        self.model = efficientNet_b3(pretrained=self.pretrained)
        self.fc = nn.Sequential(nn.Dropout(p=0.3, inplace=True),
                               nn.Linear(1536, 256),
                               nn.Linear(256, num_labels))
        self.model.classifier = self.fc

    def forward(self, x):
        return self.model(x)
    
            
class EfficientNetB4(nn.Module):
    def __init__(self, num_labels=10, pretrained=False):
        super(EfficientNetB4, self).__init__()
        self.num_labels = num_labels
        self.pretrained = pretrained
        
        self.model = efficientNet_b4(pretrained=self.pretrained)
        self.fc = nn.Sequential(nn.Dropout(p=0.4, inplace=True),
                               nn.Linear(1792, 256),
                               nn.Linear(256, self.num_labels))
        self.model.classifier = self.fc

    def forward(self, x):
        return self.model(x)
            
        
class InceptionV3(nn.Module):
    def __init__(self, num_labels=10, pretrained=False):
        super(InceptionV3, self).__init__()
        self.num_labels = num_labels
        self.pretrained = pretrained
        
        self.model = Inception_v3(pretrained=False)
        self.fc = nn.Sequential(nn.Linear(2048, 512),
                               nn.Linear(512, self.num_labels))
        self.model.fc = self.fc

    def forward(self, x):
        return self.model(x)
            

# model = efficientNet b3, b4, resNet-50, Inception v3
