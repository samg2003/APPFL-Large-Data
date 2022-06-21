import os
import time
import numpy as np
import pandas as pd
import csv
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms

from appfl.config import *
from appfl.misc.data import Dataset
from appfl.misc.utils import *  #load_model, set_seed, validation
import appfl.run_serial as rs
import appfl.run_mpi as rm

from mpi4py import MPI

import argparse 

parser = argparse.ArgumentParser()

# TODO: argument parser. Fixing constants as of now
parser.add_argument("--device", type=str, default="cpu")

args = parser.parse_args()

if torch.cuda.is_available():
    args.device = "cuda"

train_data = pd.read_csv('./Data/CheXpert-v1.0-small/train.csv')
test_data = pd.read_csv('./Data/CheXpert-v1.0-small/valid.csv')

train_data = train_data[train_data['Path'].str.contains("frontal")]
test_data = test_data[test_data['Path'].str.contains("frontal")]

train_data.to_csv('./Data/train.csv', index = False)
test_data.to_csv('./Data/test.csv', index = False)

train_data_path = "./Data/train.csv"
test_data_path = "./Data/test.csv"

num_output = 14     #number of abnormalities being trained

#Taken from CheXPert site.
class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']



#Inheriting DataSet class which takes CSV file 
class FromCSVDataset(Dataset):
    def __init__(self, path):
        xrays = []
        labels = []
        with open(path) as paths_file:
            csv_ = csv.reader(paths_file)
            next(csv_, None)
            for line in csv_:
                xray_path = line[0] 
                label = line[5:]
                #assuming uncertain or not reported data as negative
                for idx, label_ in enumerate(label):
                    if label_: label[idx] = float(label_) 
                    if not label_ or  float(label_) == -1: label[idx] = 0
                xrays.append(os.getcwd() + "/Data/" + xray_path)
                labels.append(label)
        self.xrays = xrays
        self.labels = labels
    
    def __getitem__(self, idx):
        
        transforms_list = []
        transforms_list.append(transforms.Resize((224, 224)))  #DenseNet takes 224*224 images
        transforms_list.append(transforms.ToTensor())
        transform = transforms.Compose(transforms_list)

        xray_path = self.xrays[idx]
        xray = transform(Image.open(xray_path).convert('RGB'))       #Although grayscale, DenseNet takes 3 channel
        label = torch.FloatTensor(self.labels[idx])
        
        return xray, label

    def __len__(self):
        return len(self.xrays)

a = FromCSVDataset(train_data_path)
print(a[9])