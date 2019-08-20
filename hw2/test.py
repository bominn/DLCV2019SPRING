import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
from models import Yolov1_vgg16bn, vgg19_bn
from yololoss import yoloLoss
from dataset import yolodataset, testdataset
from predict import decoder
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('test_file')
parser.add_argument('pred_file')
a = parser.parse_args()


def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    #optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict()}
             #'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

class_num = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court',
                        'ground-track-field', 'harbor', 'bridge', 'small-vehicle', 'large-vehicle', 'helicopter',  
                        'roundabout', 'soccer-ball-field', 'swimming-pool', 'container-crane']

use_gpu = torch.cuda.is_available()


#test_root = 'hw2_train_val/val1500/images/'
test_root = a.test_file
#save_root = 'hw2_train_val/'
save_root = a.pred_file

if not os.path.exists(save_root):
    os.makedirs(save_root)

learning_rate = 0.001
num_epochs = 1
batch_size = 16

model = Yolov1_vgg16bn()
#model = vgg19_bn()
optimizer = torch.optim.SGD([{"params":model.parameters()}], lr=learning_rate, momentum=0.9, weight_decay=5e-4)

print('load pre-trined model')

load_checkpoint('yolo.pth', model, optimizer)
#load_checkpoint('hw2_train_val/yolo_47.pth', model, optimizer)



for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

criterion = yoloLoss(7,2,5,0.5)

if use_gpu:
    model.cuda()

model.eval()

#valid_dataset = yolodataset(root=valid_root, train=False, transform = transforms.ToTensor())
valid_dataset = testdataset(root=test_root, transform = transforms.ToTensor())
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
print('start test')
validation_loss = 0
with torch.no_grad():
    for i, (images,name) in enumerate(valid_loader):
        images = Variable(images)
        #target = Variable(target)
        if use_gpu:
            images = images.cuda()
            #target = target.cuda()
        
        pred = model(images)
        #loss = criterion(pred,target)
        #validation_loss += loss.item()
        # for report
        
        #num = '{:04}'.format(i)
        keep = decoder(pred.cpu())
        #print(len(keep))
        pred = pred.squeeze().cpu()
        f = open(save_root+ str(name[0]), 'w')
        for i in range(len(keep)):
            num_cell = keep[i][0]
            xmin, xmax = str(keep[i][1][0]), str(keep[i][1][2])
            ymin, ymax = str(keep[i][1][1]), str(keep[i][1][3])
            cofid = keep[i][1][4]
            cell_i = num_cell//7
            cell_j = num_cell%7
            value, index = torch.max(pred[cell_i,cell_j,10:],0)
            value.numpy()
        
            f.write('{} {} {} {} {} {} {} {} {} {}\n'.format(xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, class_num[index], (value.numpy()*cofid)))
        f.close()
                    
                
                
    #validation_loss /= len(valid_loader)
    #print('valid loss={}'.format(validation_loss))
    print('finish')
