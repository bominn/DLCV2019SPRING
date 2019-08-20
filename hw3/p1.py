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
    #optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            #ngf, ndf = 64
            #nc = 3
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)   #64*64*3 expect size

netG = Generator(1)

# fixed seed
manualSeed = 96
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#Load model
load_checkpoint('model/g.pth', netG)

netG.cuda()
netG.eval()

fixed_noise = torch.randn(32, nz, 1, 1).cuda()
fake = netG(fixed_noise)
torchvision.utils.save_image(fake.cpu().data, os.path.join(a.output_file, 'fig1_2.jpg'),nrow=8)

print('GAN finish')