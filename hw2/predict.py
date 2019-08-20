import torch
from torch.autograd import Variable
import torch.nn as nn

import torchvision.transforms as transforms
import cv2
import numpy as np


def decoder(pred):
    '''
    pred(tensor) shape : (1,7,7,26)

    '''
    class_num = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court',
                        'ground-track-field', 'harbor', 'bridge', 'small-vehicle', 'large-vehicle', 'helicopter',  
                        'roundabout', 'soccer-ball-field', 'swimming-pool', 'container-crane']
    image_size = 512
    pred = pred.squeeze()
    grid_num = 7
    cell_size = 512./7
    box = {}
    for i in range(grid_num):
        for j in range(grid_num):
            if pred[i,j,4] > pred[i,j,9] and pred[i,j,4] > 0.1:
                left = (cell_size*i + pred[i,j,0]*cell_size - 0.5*pred[i,j,2]*512).numpy()
                left = max(0, left)
                top = (cell_size*j + pred[i,j,1]*cell_size - 0.5*pred[i,j,3]*512).numpy()
                top = max(0,top)
                right = (cell_size*i + pred[i,j,0]*cell_size + 0.5*pred[i,j,2]*512).numpy()
                right = min(512,right)
                bottom = (cell_size*j + pred[i,j,1]*cell_size + 0.5*pred[i,j,3]*512).numpy()
                bottom = min(512,bottom)
                cof = pred[i,j,4].numpy()
                box[i*7+j] = [left, top, right, bottom, cof]

            elif pred[i,j,9] > pred[i,j,4] and pred[i,j,9] > 0.1:
                left = (cell_size*i + pred[i,j,5]*cell_size - 0.5*pred[i,j,7]*512).numpy()
                left = max(0, left)
                top = (cell_size*j + pred[i,j,6]*cell_size - 0.5*pred[i,j,8]*512).numpy()
                top = max(0,top)
                right = (cell_size*i + pred[i,j,5]*cell_size + 0.5*pred[i,j,7]*512).numpy()
                right = min(512,right)
                bottom = (cell_size*j + pred[i,j,6]*cell_size + 0.5*pred[i,j,8]*512).numpy()
                bottom = min(512,bottom)
                cof = pred[i,j,9].numpy()
                box[i*7+j] = [left, top, right, bottom, cof]

            else:
                box[i*7+j] = [0,0,0,0,0]
            #box[i*7+j] = [left.numpy(), top.numpy(), right.numpy(), bottom.numpy(), cof.numpy()]
    #z = sorted(box.items(), key = lambda d:d[1][4])
    
    keep = nms(box, 0.5)
    return keep
    '''
    for i in range(len(keep)):
        num_cell = keep[i][0]
        xmin, xmax = str(keep[i][1][0]), str(keep[i][1][2])
        ymin, ymax = str(keep[i][1][1]), str(keep[i][1][3])
        cofid = keep[i][1][4]
        cell_i = num_cell//7
        cell_j = num_cell%7
        value, index = torch.max(pred[cell_i,cell_j,10:],0)
        value.numpy()
    
        print('{} {} {} {} {} {} {} {} {} {}\n'.format(xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, class_num[index], (value.numpy()*cofid)))
    '''
def compute_iou(box1, box2):
    #print(box1)
    #print(box2)
    """
    computing IoU
    :param box: (left, top, right, bottom)
    :return: scala value of IoU
    """

        # computing area of each rectangles
    S_rec1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    S_rec2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(box1[0], box2[0])
    right_line = min(box1[2], box2[2])
    top_line = max(box1[1], box2[1])
    bottom_line = min(box1[3], box2[3])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)

def nms(boxes, threshold):
    keep = []
    z = sorted(boxes.items(), key = lambda d:d[1][4])  #sorted by confidence
    while(z[-1][1][4] > 0):     #confidence > 0

        keep.append(z[-1])
        box1 = z[-1][1][0:4]
        del z[-1]
        if len(z)==0:
            break
        for i in range(len(z)):
            if z[len(z)-1-i][1][4] == 0:   #confidence = 0
                break
            score = compute_iou(box1, z[len(z)-1-i][1][0:4])
            if score > 0.5:
                z[len(z)-1-i][1][4] = 0
                #print('zz')  
        z = sorted(z, key = lambda d:d[1][4])
        
    return keep
#k = torch.rand(1,7,7,26)
#decoder(k)
if __name__=='__main__':
    k = torch.rand(1,7,7,26)
    