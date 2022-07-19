
# importurile din fisierele mele
from dataloader import Dataset
from retele import *
from test import *
from train import *

import torch
import torchvision # contine data loader-uri pentru seturi de date comune
import torchvision.transforms as transforms

import cv2
import matplotlib.pyplot as plt
import numpy as np


# pentru definirea neural network
import torch.nn as nn # tipuri de straturi
import torch.nn.functional as F # functii de activare

# pentru optimizatori
import torch.optim as optim

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
#device = torch.device("cuda:0" if use_cuda else "cpu")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# se creeaza o lista cu parametrii, transmisi catre constructorul DatalLoader-ului: utils.data.DataLoader
params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 1}
nr_epochs = 32 # * de aici se schimba nr de epoci

dataset = Dataset('/home/intern1/work/pytorch1/proiect_indian_food/Food Classification/', color_base='brg', zoom = True, rotate = True) # * de aici se contoleaza baza
# split in train and test. Se foloseste ca argument dataset-ul creat anterioir
nr_train = int(len(dataset)*0.8)
nr_test = int(len(dataset) - nr_train)
training_set, test_set = torch.utils.data.random_split(dataset, [nr_train, nr_test])


# se instantiaza generatorii:
training_generator = torch.utils.data.DataLoader(training_set, **params)
test_generator = torch.utils.data.DataLoader(test_set, **params)

########################################################################################

classes = ('burger', 'butter_naan', 'chai', 'chapati',
           'chole_bhature', 'dal_makhani', 'dhokla', 
           'fried_rice', 'idli', 'jalebi', 'kaathi_rolls',
           'kadai_paneer', 'kulfi', 'masala_dosa',
           'momos', 'paani_puri', 'pakode', 'pav_bhaji',
           'pizza', 'samosa')

dataiter = iter(training_generator)
images, labels = dataiter.next()


##################################### RETEAUA ###############################################


net = Net10() # * de aici schimb arhitectura

###################################### TRAIN ################################################

net.to(device)
SAVE_PATH = '/home/intern1/work/pytorch1/proiect_indian_food/modele_resize/model12.pth' # * de aici schimb path-ul

net = train_network(net, device, training_generator, test_generator, nr_epochs, SAVE_PATH)

###################################### TEST ################################################

test_network(net, test_generator, device, classes)









'''
cv2.waitKey()
cv2.destroyAllWindows()
'''







