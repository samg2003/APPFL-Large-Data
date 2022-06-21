import os
import time
import numpy as np
import pandas as pd
import csv
from PIL import Image

import torch
import torchvision
from torchvision.transforms import ToTensor

from appfl.config import *
from appfl.misc.data import *   #Dataset
from appfl.misc.utils import *  #load_model, set_seed, validation
import appfl.run_serial as rs
import appfl.run_mpi as rm

from mpi4py import MPI

import argparse 


# TODO: argument parser. Fixing constants as of now

training_data = pd.read_csv('./Data/CheXpert-v1.0-small/train.csv')
test_data = pd.read_csv('./Data/CheXpert-v1.0-small/valid.csv')

training_data = training_data[training_data['Path'].str.contains("frontal")]
test_data = test_data[test_data['Path'].str.contains("frontal")]

training_data.to_csv('./Data/train.csv', index = False)
test_data.to_csv('./Data/test.csv', index = False)

training_path = "./Data/train.csv"
testing_path = "./Data/test.csv"

