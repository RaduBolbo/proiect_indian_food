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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# se creeaza o lista cu parametrii, transmisi catre constructorul DatalLoader-ului: utils.data.DataLoader
params = {'batch_size': 16,
          'shuffle': True,
          'num_workers': 1}
nr_epochs = 2
# se creaza un obiect dataset

#dataset = torch.utils.data.Dataset('Food Classification', 'hsv')
dataset = Dataset('/home/intern1/work/pytorch1/proiect_indian_food/Food Classification/', color_base='hsv')
# split in train and test. Se foloseste ca argument dataset-ul creat anterioir
nr_train = int(len(dataset)*0.8)
nr_test = int(len(dataset) - nr_train)
training_set, test_set = torch.utils.data.random_split(dataset, [nr_train, nr_test])
# SAU:
#torch.utils.data.random_split(range(10), [2, 8], CEVA????)



# se instantiaza generatorii:
training_generator = torch.utils.data.DataLoader(training_set, **params)
test_generator = torch.utils.data.DataLoader(test_set, **params)

#dataset.afisare()

########################################################################################

classes = ('burger', 'butter_naan', 'chai', 'chapati',
           'chloe_bhature', 'dal_makhani', 'dhokla', 
           'fried_rice', 'idli', 'jalebi', 'kaathi_rolls',
           'kadai_paneer', 'kulfi', 'masala_dosa',
           'momos', 'paani_puri', 'pakode', 'pav_bhaji',
           'pizza', 'samosa')



SAVE_PATH = '/home/intern1/work/pytorch1/proiect_indian_food/modele_resize/model4.pth' # * de aiic se schimba path-ul

#net = torch.load(SAVE_PATH)
net = Net2() # * de aici se schimba reteaua
net.load_state_dict(torch.load(SAVE_PATH)) # se incarca, in acea instanta, modelul salvat, cu .load_state_dict


test_network(net, test_generator, device, classes)

