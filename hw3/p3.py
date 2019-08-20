import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
import glob
from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('target_root')
parser.add_argument('target')
parser.add_argument('output_file')

a = parser.parse_args()

#for digit dataset 
class minstm_data(Dataset):
    def __init__(self, root, train, transform):
        self.root = root
        self.transform = transform
        self.train = train
        if self.train:
            self.fname = sorted(glob.glob(root+'train/'+'*.png'))
            
        else:
            self.fname = sorted(glob.glob(os.path.join(root,'*.png')))
            
    def __len__(self):
        return len(self.fname)
    def __getitem__(self, idx):
        img = Image.open(self.fname[idx]).convert('RGB')
        name = os.path.basename(self.fname[idx])
        img = self.transform(img)
       
        return img, name

#DANN model
class DANNet(nn.Module):
    def __init__(self):
        super(DANNet, self).__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(3*3*128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Linear(128, 10)
        self.do1 = nn.Sequential(
            nn.Linear(3*3*128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5)
        )
        self.do2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5)
        )
        self.do3 = nn.Sequential(
            nn.Linear(128,1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        #print(x.size())
        x = x.view(-1, 3*3*128)
        digit = self.fc1(x)
        digit = self.fc2(digit)
        digit = self.fc3(digit)
        domain = self.do1(x)
        domain = self.do2(domain)
        domain = self.do3(domain)
        return digit, domain


img_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean, std, inplace=False)
])

# Dataloadaer
test = minstm_data(a.target_root, train = False, transform=img_transform)
test_loader = DataLoader(test, batch_size = 128, shuffle=False)

#Load model

if a.target == 'svhn':
    path = 'model/mnistm2svhn_121.pth'
elif a.target == 'usps':
    path = 'model/svhn2usps_16.pth'
elif a.target == 'mnistm':
    path = 'model/usps2mnistm_36.pth'
else:
    print('No target')

model = DANNet()
state = torch.load(path)
model.load_state_dict(state['state_dict'])
model.cuda()

#test, classify target doamin
ans = open(a.output_file,'w')
ans.write('image_name,label\n')
img_name = []
pred = []
model.eval()
with torch.no_grad():
    for idx, (img,name) in enumerate(test_loader):
        out_tar, _ = model(img.cuda())
        out = torch.max(out_tar,1)[1]
        out = out.cpu().numpy()
        for i in range(len(name)):
            img_name.append(name[i])
            pred.append(out[i])
for i in range(len(img_name)):
    ans.write('{},{}\n'.format(img_name[i],str(pred[i])))
ans.close()

print('DANN finish')