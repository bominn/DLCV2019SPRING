import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
import os
import sys
import random
import glob

num = [7522, 15347, 3010, 498, 2936, 608, 612, 6940, 1976, 40371, 13505, 250, 524, 581, 1766, 94]
class yolodataset(Dataset):                                                                                                   
    def __init__(self, root, train, transform):
        print('init data')
        self.root = root
        self.train = train
        self.transform = transform
        self.fnames = []
        self.image_size = 448
        self.classnum = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court',
                        'ground-track-field', 'harbor', 'bridge', 'small-vehicle', 'large-vehicle', 'helicopter',  
                        'roundabout', 'soccer-ball-field', 'swimming-pool', 'container-crane']
        self.mean = (123,117,104) #RGB

        if self.train:
            for i in range(15000):
                self.fnames.append('{:05}'.format(i))
        else:
            for i in range(1500):
                self.fnames.append('{:04}'.format(i))
        
    def __getitem__(self, idx): 
        img = cv2.imread(self.root+'\\images\\'+self.fnames[idx]+'.jpg')
        label = open(self.root+'\\labelTxt_hbb\\'+self.fnames[idx]+'.txt')
        
        label = label.readlines()
        label = [l.rstrip('\n') for l in label]
        boxes = []
        all_label = []
        for i in range(len(label)):
            tmp = label[i].split(' ')
            xmin, ymin = float(tmp[0]), float(tmp[1])
            xmax, ymax = float(tmp[4]), float(tmp[5])
            boxes.append([xmin,ymin,xmax,ymax])
            all_label.append(self.classnum.index(tmp[8])+10)
        boxes = torch.Tensor(boxes)
        all_label = torch.LongTensor(all_label)
        #print(boxes)
        
        if self.train:
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.randomScale(img, boxes)
            #img = self.randomBlur(img)
            #img = self.RandomBrightness(img)
            #img = self.RandomHue(img)
            #img = self.RandomSaturation(img)
            img,boxes,all_label = self.randomShift(img,boxes,all_label)
            #img,boxes,all_label = self.randomCrop(img,boxes,all_label)
        #print(boxes)
        
        img = self.BGR2RGB(img)
        #img = self.subMean(img,self.mean)
        #img = img/255
        img = cv2.resize(img,(self.image_size, self.image_size))
        #print(img.shape)
        img = self.transform(img)
        #print(img.size())
        #print(boxes)
        #print(all_label)
        target = self.encoder(boxes, all_label) #7*7*26

        return img, target

    def __len__(self):
        if self.train:
            return 15000
        else:
            return 1500

    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    def BGR2HSV(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    def HSV2BGR(self,img):
        return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)

    def subMean(self,bgr,mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr
    
    def encoder(self, box, label):
        #original image size = 512*512
        grid_num = 7
        target = torch.zeros((grid_num,grid_num,26))
        cell_size = 512./grid_num
        
        for i in range(len(label)):
            tmp = box[i]
            xmin, ymin = float(tmp[0]), float(tmp[1])
            xmax, ymax = float(tmp[2]), float(tmp[3])
            xc, yc = (xmax+xmin)/2, (ymax+ymin)/2

            center_x, center_y = int(xc//cell_size), int(yc//cell_size)
            #print(center_x)
            #print(center_y)
            # box1
            target[center_x, center_y, 0] = (xc%cell_size)/cell_size
            target[center_x, center_y, 1] = (yc%cell_size)/cell_size
            target[center_x, center_y, 2] = (xmax-xmin)/512
            target[center_x, center_y, 3] = (ymax-ymin)/512
            target[center_x, center_y, 4] = 1
            # box2
            target[center_x, center_y, 5] = (xc%cell_size)/cell_size
            target[center_x, center_y, 6] = (yc%cell_size)/cell_size
            target[center_x, center_y, 7] = (xmax-xmin)/512
            target[center_x, center_y, 8] = (ymax-ymin)/512
            target[center_x, center_y, 9] = 1

            #class
            target[center_x, center_y, label[i]] = 1 
                        
        return target

    def randomCrop(self,bgr,boxes,labels):
        if random.random() < 0.5:
            #print("?")
            center = (boxes[:,2:]+boxes[:,:2])/2
            height,width,c = bgr.shape
            h = random.uniform(0.85*height,height)
            w = random.uniform(0.85*width,width)
            x = random.uniform(0,width-w)
            y = random.uniform(0,height-h)
            x,y,h,w = int(x),int(y),int(h),int(w)

            center = center - torch.FloatTensor([[x,y]]).expand_as(center)
            mask1 = (center[:,0]>0) & (center[:,0]<w)
            mask2 = (center[:,1]>0) & (center[:,1]<h)
            mask = (mask1 & mask2).view(-1,1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if(len(boxes_in)==0):
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[x,y,x,y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:,0]=boxes_in[:,0].clamp_(min=0,max=w)
            boxes_in[:,2]=boxes_in[:,2].clamp_(min=0,max=w)
            boxes_in[:,1]=boxes_in[:,1].clamp_(min=0,max=h)
            boxes_in[:,3]=boxes_in[:,3].clamp_(min=0,max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y+h,x:x+w,:]
            for i in range(boxes_in.size()[0]):
                if boxes_in[i][2] > 512.0 or boxes_in[i][3] > 512.0:
                    return bgr, boxes, labels
            return img_croped,boxes_in,labels_in
        return bgr,boxes,labels

    def randomShift(self,bgr,boxes,labels):
        #平移变换
        center = (boxes[:,2:]+boxes[:,:2])/2
        if random.random() <0.5:
            height,width,c = bgr.shape
            after_shfit_image = np.zeros((height,width,c),dtype=bgr.dtype)
            after_shfit_image[:,:,:] = (104,117,123) #bgr
            shift_x = random.uniform(-width*0.1,width*0.1)
            shift_y = random.uniform(-height*0.1,height*0.1)
            #print(bgr.shape,shift_x,shift_y)
            #原图像的平移
            if shift_x>=0 and shift_y>=0:
                after_shfit_image[int(shift_y):,int(shift_x):,:] = bgr[:height-int(shift_y),:width-int(shift_x),:]
            elif shift_x>=0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),int(shift_x):,:] = bgr[-int(shift_y):,:width-int(shift_x),:]
            elif shift_x <0 and shift_y >=0:
                after_shfit_image[int(shift_y):,:width+int(shift_x),:] = bgr[:height-int(shift_y),-int(shift_x):,:]
            elif shift_x<0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),:width+int(shift_x),:] = bgr[-int(shift_y):,-int(shift_x):,:]

            shift_xy = torch.FloatTensor([[int(shift_x),int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:,0] >0) & (center[:,0] < width)
            mask2 = (center[:,1] >0) & (center[:,1] < height)
            mask = (mask1 & mask2).view(-1,1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if len(boxes_in) == 0:
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[int(shift_x),int(shift_y),int(shift_x),int(shift_y)]]).expand_as(boxes_in)
            boxes_in = boxes_in+box_shift
            #print(mask.view(-1))
            for i in range(boxes_in.size()[0]):
                if boxes_in[i][2] > 512.0 or boxes_in[i][3] > 512.0:
                    return bgr, boxes, labels
            labels_in = labels[mask.view(-1)]
            return after_shfit_image,boxes_in,labels_in
        return bgr,boxes,labels

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h,w,_ = im.shape
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
            return im_lr, boxes
        return im, boxes

    def randomScale(self,bgr,boxes):

        if random.random() < 0.5:
            scale = random.uniform(0.9,1.1)
            height,width,c = bgr.shape
            bgr = cv2.resize(bgr,(int(width*scale),int(height*scale)))
            scale_tensor = torch.FloatTensor([[scale,scale,scale,scale]]).expand_as(boxes)
            boxes_scale = boxes * scale_tensor
            for i in range(boxes.size()[0]):
                if boxes_scale[i][2] > 512.0 or boxes_scale[i][3] > 512.0:
                    return bgr, boxes
            return bgr,boxes_scale
        return bgr,boxes

    def randomBlur(self,bgr):
        if random.random()<0.5:
            bgr = cv2.blur(bgr,(5,5))
        return bgr

    def RandomBrightness(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.75,1.25])
            v = v*adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr   

    def RandomHue(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.75,1.25])
            h = h*adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    
    def RandomSaturation(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.75,1.25])
            s = s*adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

class testdataset(Dataset):
    def __init__(self, root, transform):
        print('init data')
        self.root = root
        self.transform = transform
        self.fnames = sorted(glob.glob(root+'*.jpg'))
        self.image_size = 448
        self.classnum = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court',
                        'ground-track-field', 'harbor', 'bridge', 'small-vehicle', 'large-vehicle', 'helicopter',  
                        'roundabout', 'soccer-ball-field', 'swimming-pool', 'container-crane']
        self.mean = (123,117,104) #RGB

        #print(self.len)
     
        

    def __getitem__(self, idx): 
        img = cv2.imread(self.fnames[idx])
        name = os.path.basename(self.fnames[idx]).replace('jpg','txt')
        
        img = self.BGR2RGB(img)
        img = cv2.resize(img,(self.image_size, self.image_size))
        img = self.transform(img)

        return img, name

    def __len__(self):
        return len(self.fnames)
        
    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def main():
    test_root = 'hw2_train_val/val1500/images/'
    test_dataset = testdataset(root=test_root, transform = transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    for idc, (image,name) in enumerate(test_loader):
        if idc == 0:
            print(type(image))
            break
            
    print('finish') 
    '''
    train_root = 'hw2_train_val\\train15000'
    train_dataset = yolodataset(root=train_root, train=True, transform = transforms.ToTensor())
    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=False)
    count = [0]*16
    
    print(count)
    for idc, (image, label) in enumerate(train_loader):
        #if idc ==0:
            #print(label)
         #   break
    #image, label = next(iter(train_loader))
    
        if (idc+1)%500 == 0:
            print(idc+1)
        for i in range(7):
            for j in range(7):
                temp = label[0][i][j][10:]
                #print(temp)
                for k in range(len(temp)):
                    if temp[k] != 0:
                        count[k]+=1
    print(count)
    '''
if __name__ == '__main__':
    #image = cv2.imread('hw2_train_val\\train15000\\images\\00000.jpg')
    #print(image)
    main()