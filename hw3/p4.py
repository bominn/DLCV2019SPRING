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
        img = Image.open(self.fname[idx]).convert('L')
        name =  os.path.basename(self.fname[idx])
        img = img.convert('RGB')
        img = self.transform(img)
        
        return img, name

img_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    #transforms.Normalize(mean, std, inplace=False)
])
test = minstm_data(a.target_root, train = False, transform=img_transform)
test_loader = DataLoader(test, batch_size = 128, shuffle=False)

#Load model
if a.target == 'svhn':
    path = 'model/mnistm2svhn_vae'
    cls_path = 'model/mnistm2svhn_dnet'
elif a.target == 'usps':
    path = 'model/svhn2usps_vae'
    cls_path = 'model/svhn2usps_dnet'
elif a.target == 'mnistm':
    path = 'model/usps2mnistm_vae'
    cls_path = 'model/usps2mnistm_dnet'
else:
    print('No target')

vae = torch.load(path)
d_digit = torch.load(cls_path)

vae.cuda()
d_digit.cuda()
vae.eval()
d_digit.eval()

#test classify target doamin
ans = open(a.output_file,'w')
ans.write('image_name,label\n')
img_name = []
p = []
for idx, (img,name) in enumerate(test_loader):
    test_batch = img.cuda()
    pred = d_digit(vae((test_batch*2-1),return_enc=True))
    pred = torch.max(pred,1)[1] 
    pred = pred.cpu().numpy()
    
    for i in range(len(name)):
        img_name.append(name[i])
        p.append(pred[i])
    #ans.write('{},{}'.format(name[0],str(pred[0])))
    
for k in range(len(img_name)):
    ans.write('{},{}\n'.format(img_name[k],str(p[k])))
ans.close()

print('Improved DANN finish')