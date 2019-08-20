import numpy as np
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('output_file')

a = parser.parse_args()

if not os.path.exists(a.output_file):
    os.makedirs(a.output_file)

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)

class Generator(nn.Module):
    def __init__(self, figsize=64):
        super(Generator, self).__init__()
        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( 101, figsize * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(figsize * 8),
            nn.ReLU(inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(figsize * 8, figsize * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(figsize * 4),
            nn.ReLU(inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(figsize * 4, figsize * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(figsize * 2),
            nn.ReLU(inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(figsize * 2, figsize, 4, 2, 1, bias=False),
            nn.BatchNorm2d(figsize),
            nn.ReLU(inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(figsize, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, X):
        output = self.decoder(X)
        return output
netG = Generator()

# Fixed seed
manualSeed = 42
#print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# domain input 1=smile, 0=not smile
up = np.ones(10)
down = np.zeros(10)
fixed_class = np.hstack((up,down))
fixed_class = torch.from_numpy(fixed_class).view(20,1,1,1).float()
fixed_noise = torch.randn(10, 100, 1, 1)
fixed_noise = torch.cat((fixed_noise,fixed_noise))
fixed_input = (torch.cat((fixed_noise, fixed_class),1)).cuda()

#Load model
load_checkpoint('model/acgang.pth',netG)

#Generate image and save
netG.cuda()
netG.eval()
fixed_img_output = netG(fixed_input.cuda())
torchvision.utils.save_image(fixed_img_output.cpu().data, os.path.join(a.output_file, 'fig2_2.jpg'), nrow=10)

print('ACGAN finish')