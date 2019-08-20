from os import listdir
import os
import pandas as pd
import numpy as np
import argparse
import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import skimage

parser = argparse.ArgumentParser()
parser.add_argument('video_path')
parser.add_argument('output_folder')
ap = parser.parse_args()

def trans(image):
    transform_input = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform_input(image)

video_path = ap.video_path
category_list = sorted(listdir(video_path))

all_video_frame = []
cnn_feature_extractor = torchvision.models.densenet121(pretrained=True).features.cuda() 
with torch.no_grad():
    for category in category_list:
        print("category:",category)
        image_list_per_folder = sorted(listdir(os.path.join(video_path,category)))
        category_frames = []
        for image in image_list_per_folder:
            image_rgb = skimage.io.imread(os.path.join(video_path, category,image))
            image_nor = trans(image_rgb)
            feature = cnn_feature_extractor(image_nor.view(1,3,224,224).cuda()).cpu().view(1024*7*7)
            category_frames.append(feature)
        all_video_frame.append(torch.stack(category_frames))

video_lengths = [len(s) for s in all_video_frame]
class seq2seq(nn.Module):
    def __init__(self, input_size, hidden_size=512, n_layers=2, dropout=0.5):
        super(seq2seq, self).__init__()
        self.hidden_size =  hidden_size
        self.lstm = nn.LSTM(input_size, self.hidden_size, n_layers,
                          dropout=0.5, bidirectional=False)

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, 11),
        )
    def forward(self, padded_sequence, input_lengths, hidden=None):
        packed = torch.nn.utils.rnn.pack_padded_sequence(padded_sequence, input_lengths,batch_first=True)
        outputs, (hn,cn) = self.lstm(packed, hidden) 

        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        fc_output = self.fc2(outputs)
        return fc_output

print("load model ...")
feature_size = 1024*7*7
model = seq2seq(feature_size,hidden_size=512,dropout=0.5, n_layers=2).cuda()
model.load_state_dict(torch.load("models/seq_model.pth"))

with torch.no_grad():
    model.eval()
    valid_output = []
    for valid_X, length in zip(all_video_frame, video_lengths):
        input_valid_X = valid_X.unsqueeze(0)
        output = model(input_valid_X.cuda(), [length])
        prediction = torch.argmax(torch.squeeze(output.cpu()),1).data.numpy()
        valid_output.append(prediction)

valid_dir_name = sorted(listdir(video_path))

if not os.path.exists(ap.output_folder):
    os.makedirs(ap.output_folder)

for i in range(len(valid_dir_name)):
    with open(os.path.join(ap.output_folder, valid_dir_name[i]+'.txt'), "w") as f:
        for j, pred in enumerate(valid_output[i]):
            f.write(str(pred))
            if j != len(valid_output[i])-1:
                f.write("\n")