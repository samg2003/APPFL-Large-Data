#TODO: Haven't added COVID-19 dataset yet.
#TODO: get_data function (done)
#TODO: get_model function (done)
#TODO: main (in process)

import os
import time
import numpy as np
import pandas as pd
import csv
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import math

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

print("Total training data: ", len(train_data))
len_training_data = len(train_data)

train_data_path = "./Data/train.csv"
test_data_path = "./Data/test.csv"

num_output = 14     #number of abnormalities being trained

#Taken from CheXPert site.
class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

num_clients = 3
device = 'cpu'

#Inheriting DataSet class which takes CSV file 
class FromCSVDataset(Dataset):
    def __init__(self, path, start_line = -1, end_line = -1):
        '''
        path: path of CSV file which records location of x-rays and their labelings
        start_line: Starting line of the csv file from where data would be part of this object
        end_line:   endling line of the csv file till where data would be part of this object   

        start_line and end_line is used for segregating data for each client
        '''
        xrays = []
        labels = []
        
        with open(path) as paths_file:
            csv_ = csv.reader(paths_file)

            #skipping header line
            next(csv_, None)

            #iterating until start_line
            if start_line != -1:
                for _ in range(start_line):
                    next(csv_, None)

            num_iter_left = end_line - start_line
            for line in csv_:
                if num_iter_left <= 0 and start_line != -1: break
                num_iter_left -= 1
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


def data_processing(comm: MPI.Comm):
    #TODO: Check if it works.
    #Have to check whether MPI proctocol for data loading is necassary?

    test_dataset = FromCSVDataset(test_data_path)
    train_datasets = []

    split_size = int(len_training_data/num_clients)
    for client_idx in range(num_clients):
        train_datasets.append(FromCSVDataset(train_data_path, split_size * client_idx, split_size * (client_idx + 1)))
    
    return train_datasets, test_dataset
            

class DenseNet121(nn.Module):
    """
    DenseNet121 model with additional Sigmoid layer for classification
    TODO: See improvements in model and scope of using lateral X-rays by merging two densenet models (ensemble training).
    """
    def __init__(self, num_output):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained = False)
        num_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_features, num_output),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


def get_model(comm: MPI.Comm):
    ## User-defined model
    model = DenseNet121(num_output)
    return model


def federated_learning():
    '''
    Part of this code is from https://github.com/APPFL/APPFL/blob/96a1da6d7aeb64a8a78bfcc7dc60fa37273dfdb5/examples/mnist.py
    '''

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    '''Config'''
    cfg = OmegaConf.structured(Config)

    cfg.device = device
    cfg.reproduce = True
    if cfg.reproduce == True:
        set_seed(1)

     ## clients
    cfg.num_clients = num_clients
    cfg.fed.args.optim = 'Adam'
    cfg.fed.args.optim_args.lr = 1e-3
    cfg.fed.args.num_local_epochs = 1
    cfg.validation = False
    
    ## server
    cfg.fed.servername = "ServerFedAvg"
    cfg.num_epochs = 1

    start_time = time.time()

    """ User-defined model """
    model = get_model(comm)
    
    #maybe try BCE loss, worked better last time.
    #loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = torch.nn.BCELoss()
    
    ## loading models
    cfg.load_model = False
    if cfg.load_model == True:
        cfg.load_model_dirname = "./save_models"
        cfg.load_model_filename = "Model"
        model = load_model(cfg)

    """ User-defined data """
    train_datasets, test_dataset = data_processing(comm)

    print(
        "-------Loading_Time=",
        time.time() - start_time,
    )

    """ saving models """
    cfg.save_model = True
    if cfg.save_model == True:
        cfg.save_model_dirname = "./save_models"
        cfg.save_model_filename = "Model"

    """ Running """
    if comm_size > 1:
        if comm_rank == 0:
            rm.run_server(
                cfg, comm, model, loss_fn, num_clients, test_dataset
            )
        else:
            rm.run_client(
                cfg, comm, model, loss_fn, num_clients, train_datasets, test_dataset
            )
        print("------DONE------", comm_rank)
    else:
        rs.run_serial(cfg, model, loss_fn, train_datasets, test_dataset)


def train(model, train_dataset, test_dataset, num_epoch, save_directory):
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001,
                               betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0) 
    loss = torch.nn.BCELoss()

    ## loading models
    load_model = False
    if load_model == True:
        path_to_model = "enter file path"
        model_loaded = torch.load(path_to_model)
        model.load_state_dict(model_loaded['state_dict'])
        optimizer.load_state_dict(model_loaded['optimizer'])

    #training starts
    train_start = []
    train_end = []
    for epoch in range(num_epoch):
        train_start.append(time.time())
        
        #one epoch
        model.train()
        total_training_loss = 0
        for idx, (inp, label) in enumerate(train_dataset):
            if device == 'cuda':
                label = label.cuda(non_blocking = True)
            out = model(inp)
            loss_val = loss(out, label)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            total_training_loss += loss_val.item()
        loss_training = total_training_loss/len(train_dataset)

        #validation
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for idx, (inp, label) in enumerate(test_dataset):
                if device == 'cuda':
                    label = label.cuda(non_blocking = True)
                out = model(inp)
                
                total_test_loss += loss(out, label)
                
        loss_test = total_test_loss / len(test_dataset)

        train_end.append(time.time())

        #info print in terminal
        print("Training loss: ", str(loss_training), "Test loss: ", str(loss_test))

        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 
                            'optimizer' : optimizer.state_dict()}, 
                            save_directory + 'model_trained_' + str(epoch) + '.pth.tar')

def central_learning():
    model = DenseNet121(num_output)
    if device == 'cuda':
        model = torch.nn.DataParallel(model).cuda()

    
    train_dataset_pre = FromCSVDataset(train_data_path)
    test_dataset_pre = FromCSVDataset(test_data_path)

    train_dataset = torch.utils.data.DataLoader(dataset=train_dataset_pre, batch_size=64, shuffle=True,  num_workers=5)
    test_dataset = torch.utils.data.DataLoader(dataset=test_dataset_pre, num_workers=5)

    train(model, train_dataset, test_dataset, num_epoch= 1, save_directory = "central_model/")

if __name__ == "__main__":
    federated_learning()