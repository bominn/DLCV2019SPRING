import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
from models import Yolov1_vgg16bn, vgg19_bn
from yololoss import yoloLoss
from dataset import yolodataset
from predict import decoder
import numpy as np

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict()}
             #'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    #optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

class_num = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court',
                        'ground-track-field', 'harbor', 'bridge', 'small-vehicle', 'large-vehicle', 'helicopter',  
                        'roundabout', 'soccer-ball-field', 'swimming-pool', 'container-crane']

use_gpu = torch.cuda.is_available()

train_root = 'hw2_train_val/train15000'
valid_root = 'hw2_train_val/val1500'
save_root = 'hw2_train_val/'
learning_rate = 0.001
num_epochs = 20
batch_size = 16

#model = Yolov1_vgg16bn()
model = vgg19_bn()
optimizer = torch.optim.SGD([{"params":model.parameters()}], lr=learning_rate, momentum=0.9, weight_decay=5e-4)

print('load pre-trined model')
'''
#vgg = models.vgg16_bn(pretrained=True)
vgg = models.vgg19_bn(pretrained=True)
new_state_dict = vgg.state_dict()
dd = model.state_dict()
for k in new_state_dict.keys():
    #print(k)
    if k in dd.keys() and k.startswith('features'):
        #print('yes')
        dd[k] = new_state_dict[k]
model.load_state_dict(dd)

# test load model
'''
load_checkpoint('hw2_train_val/model_19_50.pth', model, optimizer)
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

#print(model)

criterion = yoloLoss(7,2,5,0.5)

if use_gpu:
    model.cuda()

model.train()



train_dataset = yolodataset(root=train_root, train=True, transform = transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

valid_dataset = yolodataset(root=valid_root, train=False, transform = transforms.ToTensor())
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
best_test_loss = 4.8

print('start train')

for epoch in range(num_epochs):
    model.train()
    # if epoch == 1:
    #     learning_rate = 0.0005
    # if epoch == 2:
    #     learning_rate = 0.00075
    # if epoch == 3:
    #     learning_rate = 0.001
    
    num_iter = 0
    if epoch > 40:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    total_loss = 0.
    
    for i,(images,target) in enumerate(train_loader):
        images = Variable(images)
        target = Variable(target)
        if use_gpu:
            images,target = images.cuda(),target.cuda()
        
        pred = model(images)
        loss = criterion(pred,target)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 50 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' 
            %(epoch+1, num_epochs, i+1, len(train_loader), loss.item(), total_loss / (i+1)))
            num_iter += 1
        
    validation_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i,(images,target) in enumerate(valid_loader):
            images = Variable(images)
            target = Variable(target)
            if use_gpu:
                images,target = images.cuda(),target.cuda()
            
            pred = model(images)
            loss = criterion(pred,target)
            validation_loss += loss.item()
            # for report
            
            if epoch == 0 or epoch == 19 or epoch ==39:
                if i==76 or i==86 or i==907:
                    num = '{:04}'.format(i)
                    keep = decoder(pred.cpu())
                    #print(len(keep))
                    pred = pred.squeeze().cpu()
                    f = open('hw2_train_val/val1500/label_19_'+str(epoch+1)+'\\'+num+'.txt', 'w')
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
            
                
                
    validation_loss /= len(valid_loader)
    print('valid loss={}'.format(validation_loss))
    
    if best_test_loss > validation_loss:
        best_test_loss = validation_loss
        print('get best test loss %.5f' % best_test_loss)
        path = save_root+'model_19_'+str(epoch++50+1)+'.pth'
        save_checkpoint(path, model, optimizer)
        #torch.save(model.state_dict(),'best.pth')
path = save_root+'best_6.pth'
save_checkpoint(path, model, optimizer)         
#torch.save(model.state_dict(),'yolo.pth')
print('finish')