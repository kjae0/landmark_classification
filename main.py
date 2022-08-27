import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader as DataLoader
from torch.utils.data import Dataset as Dataset

import torchvision
from torchvision.transforms import transforms as transforms

import os
import numpy
import time
import pandas as pd
import matplotlib.pyplot as plt

import custom_dataset
import models

from sklearn.model_selection import train_test_split

data_dir = "../landmark_classification/dataset/"
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
inception_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])
effb3_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])
effb4_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor()
])


# normalize

resnet_dataset = custom_dataset.Dataset(os.path.join(data_dir, 'train'), 
                                        os.path.join(data_dir, 'train.csv'),
                                        transform=resnet_transform)
inception_dataset = custom_dataset.Dataset(os.path.join(data_dir, 'train'), 
                                        os.path.join(data_dir, 'train.csv'),
                                        transform=inception_transform)
effb3_dataset = custom_dataset.Dataset(os.path.join(data_dir, 'train'), 
                                        os.path.join(data_dir, 'train.csv'),
                                        transform=effb3_transform)
effb4_dataset = custom_dataset.Dataset(os.path.join(data_dir, 'train'), 
                                        os.path.join(data_dir, 'train.csv'),
                                        transform=effb4_transform)

def show_images(images, image_per_row=10):
    images, _ = images
    print(_)
    plt.figure(figsize=(20, 10))
    plt.axis('off')
    plt.imshow(torchvision.utils.make_grid(images, nrow=image_per_row).permute(1, 2, 0))
    
# hyperparameter setting
batch_size = 12
learning_rate = 1e-4
n_epochs = 10
random_seed = 42
test_size = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

resnet_train, resnet_val = train_test_split(resnet_dataset,
                                            test_size=test_size,
                                            stratify=resnet_dataset.labels,
                                            random_state=random_seed)
resnet_train_dataloader = DataLoader(resnet_train, 
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=False)
resnet_val_dataloader = DataLoader(resnet_val, 
                              batch_size=batch_size,
                              shuffle=False,
                              drop_last=False)

inception_train, inception_val = train_test_split(inception_dataset, 
                                                  test_size=test_size, 
                                                  stratify=inception_dataset.labels,
                                                  random_state=random_seed)
inception_train_dataloader = DataLoader(inception_train, 
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=False)
inception_val_dataloader = DataLoader(inception_val, 
                              batch_size=batch_size,
                              shuffle=False,
                              drop_last=False)

effb3_train, effb3_val = train_test_split(effb3_dataset, 
                                          test_size=test_size,
                                          stratify=effb3_dataset.labels,
                                          random_state=random_seed)
effb3_train_dataloader = DataLoader(effb3_train, 
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=False)
effb3_val_dataloader = DataLoader(effb3_val, 
                              batch_size=batch_size,
                              shuffle=False,
                              drop_last=False)

effb4_train, effb4_val = train_test_split(effb4_dataset, 
                                          test_size=test_size, 
                                          stratify=effb4_dataset.labels,
                                          random_state=random_seed)
effb4_train_dataloader = DataLoader(effb4_train, 
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=False)
effb4_val_dataloader = DataLoader(effb4_val, 
                              batch_size=batch_size,
                              shuffle=False,
                              drop_last=False)

print(f'train dataset size is {len(resnet_train)}')
print(f'test dataset size is {len(resnet_val)}')

resnet = models.ResNet50().to(device)
resnet = torch.nn.DataParallel(resnet).to(device)
# 224 224
inception = models.InceptionV3().to(device)
inception = torch.nn.DataParallel(inception).to(device)
# 299 299
effb3 = models.EfficientNetB3().to(device)
effb3 = torch.nn.DataParallel(effb3).to(device)
# 300 300
effb4 = models.EfficientNetB4().to(device)
effb4 = torch.nn.DataParallel(effb4).to(device)
# 380 380

resnet_optimizer = optim.Adam(resnet.parameters(), lr=learning_rate)
resnet_scheduler = optim.lr_scheduler.CosineAnnealingLR(resnet_optimizer, T_max=20, eta_min=0, )

inception_optimizer = optim.Adam(inception.parameters(), lr=learning_rate)
inception_scheduler = optim.lr_scheduler.CosineAnnealingLR(inception_optimizer, T_max=20, eta_min=0, )

effb3_optimizer = optim.Adam(effb3.parameters(), lr=learning_rate)
effb3_scheduler = optim.lr_scheduler.CosineAnnealingLR(effb3_optimizer, T_max=20, eta_min=0, )

effb4_optimizer = optim.Adam(effb4.parameters(), lr=learning_rate)
effb4_scheduler = optim.lr_scheduler.CosineAnnealingLR(effb4_optimizer, T_max=20, eta_min=0, )

criterion = nn.CrossEntropyLoss().to(device)

def getAccuracy(model, test, dataset_size):
    correct = 0
    for x, y in test:
        x = x.to(device)
        y = y.to(device)
        prediction = model(x)
        check = torch.argmax(prediction, dim=1)==y
        correct += torch.sum(check)

    correct = correct.detach()
    return correct / dataset_size

def saveModel(model, save_dir, name, epoch):
    save_dir = os.path.join(save_dir, name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, f'{epoch+1}.pt'))
    
def saveFigure(plot, save_dir, name):
    save_dir = os.path.join(save_dir, 'figure')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.plot([i for i in range(len(plot))], plot)
    plt.savefig(os.path.join(save_dir, name), facecolor='white')
    
print('training resnet 50 model...')
s = time.time()
    
save_dir = ""
# resnet
# about 7~8 minutes per epoch (setting:cpu)
resnet_train_acc_lst = []
resnet_val_acc_lst = []
for epoch in range(n_epochs):
    resnet.train()
    losses = 0
    n_iter = 0
    for x, y in resnet_train_dataloader:
        x = x.to(device)
        y = y.to(device)
        
        hypothesis = resnet(x)
        loss = criterion(hypothesis, y)
        
        losses += loss
        n_iter += 1
        
        resnet_optimizer.zero_grad()
        loss.backward()
        resnet_optimizer.step()
        
    print(f'{epoch+1}/{n_epochs} : {losses/n_iter}')
    
    resnet.eval()
    with torch.no_grad():
        train_acc = getAccuracy(resnet, resnet_train_dataloader, len(resnet_train))
        val_acc = getAccuracy(resnet, resnet_val_dataloader, len(resnet_val))
        resnet_train_acc_lst.append(train_acc.detach())
        resnet_val_acc_lst.append(val_acc.detach())
    print(f'train accuracy : {train_acc*100}%, validation accuracy : {val_acc*100}%')
    if (epoch+1)%10==0:
        saveModel(resnet, save_dir, 'resnet50', epoch)
    resnet_scheduler.step()

saveFigure(resnet_train_acc_lst, save_dir, 'resnet_train_accuracy.png')
saveFigure(resnet_val_acc_lst, save_dir, 'resnet_val_accuracy.png')

print(f'running time : {time.time() - s}s\n\n')


print('training inception v3 model...')
s = time.time()
# inception
# about 9~10 minutes per epoch (setting:cpu)
inception_train_acc_lst = []
inception_val_acc_lst = []
for epoch in range(n_epochs):
    inception.train()
    losses = 0
    n_iter = 0
    for x, y in inception_train_dataloader:
        x = x.to(device)
        y = y.to(device)
        
        hypothesis, _ = inception(x)
        loss = criterion(hypothesis, y)
        
        losses += loss
        n_iter += 1
        
        inception_optimizer.zero_grad()
        loss.backward()
        inception_optimizer.step()
        
    print(f'{epoch+1}/{n_epochs} : {losses/n_iter}')
    
    inception.eval()
    with torch.no_grad():
        train_acc = getAccuracy(inception, inception_train_dataloader, len(inception_train))
        val_acc = getAccuracy(inception, inception_val_dataloader, len(inception_val))
        inception_train_acc_lst.append(train_acc.detach())
        inception_val_acc_lst.append(val_acc.detach())
    print(f'train accuracy : {train_acc*100}%, validation accuracy : {val_acc*100}%')
    if (epoch+1)%10==0:
        saveModel(inception, save_dir,'inception_v3', epoch)

saveFigure(inception_train_acc_lst, save_dir, 'inception_train_accuracy.png')
saveFigure(inception_val_acc_lst, save_dir, 'inception_val_accuracy.png')
print(f'running time : {time.time() - s}s\n\n')

print('training efficientNet b3 model...')
s = time.time()
# efficientNet b3
# about 5 minutes per epoch (setting:cpu)
effb3_train_acc_lst = []
effb3_val_acc_lst = []
for epoch in range(n_epochs):
    losses = 0
    n_iter = 0
    for x, y in effb3_train_dataloader:
        effb3.train()
        x = x.to(device)
        y = y.to(device)
        
        hypothesis = effb3(x)
        loss = criterion(hypothesis, y)
        
        losses += loss
        n_iter += 1
        
        effb3_optimizer.zero_grad()
        loss.backward()
        effb3_optimizer.step()
        
    print(f'{epoch+1}/{n_epochs} : {losses/n_iter}')
    
    effb3.eval()
    with torch.no_grad():
        train_acc = getAccuracy(effb3, effb3_train_dataloader, len(effb3_train))
        val_acc = getAccuracy(effb3, effb3_val_dataloader, len(effb3_val))
        effb3_train_acc_lst.append(train_acc.detach())
        effb3_val_acc_lst.append(val_acc.detach())
    print(f'train accuracy : {train_acc}, validation accuracy : {val_acc}')
    if (epoch+1)%10==0:
        saveModel(effb3, save_dir,'efficientNet_b3', epoch)

saveFigure(effb3_train_acc_lst, save_dir, 'effb3_train_accuracy.png')
saveFigure(effb3_val_acc_lst, save_dir, 'effb3_val_accuracy.png')
print(f'running time : {time.time() - s}s\n\n')


print('training efficientNet b4 model...')
s = time.time()
# inception
# about 5 minutes per epoch (setting:cpu)
effb4_train_acc_lst = []
effb4_val_acc_lst = []
for epoch in range(n_epochs):
    losses = 0
    n_iter = 0
    for x, y in effb4_train_dataloader:
        x = x.to(device)
        y = y.to(device)
        
        hypothesis = effb4(x)
        loss = criterion(hypothesis, y)
        
        losses += loss
        n_iter += 1
        
        effb4_optimizer.zero_grad()
        loss.backward()
        effb4_optimizer.step()
        break
        
    print(f'{epoch+1}/{n_epochs} : {losses/n_iter}')
    train_acc = getAccuracy(effb4, effb4_train_dataloader, len(effb4_train))
    val_acc = getAccuracy(effb4, effb4_val_dataloader, len(effb4_val))
    effb4_train_acc_lst.append(train_acc.detach())
    effb4_val_acc_lst.append(val_acc.detach())
    print(f'train accuracy : {train_acc*100}%, validation accuracy : {val_acc*100}%')
    if (epoch+1)%10==0:
        saveModel(effb4, save_dir,'efficientNet_b4', epoch)

saveFigure(effb4_train_acc_lst, save_dir, 'effb4_train_accuracy.png')
saveFigure(effb4_val_acc_lst, save_dir, 'effb4_val_accuracy.png')
print(f'running time : {time.time() - s}s\n\n')


