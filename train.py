
import torch
from test import *

from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary

# pentru definirea neural network
import torch.nn as nn # tipuri de straturi
import torch.nn.functional as F # functii de activare

# pentru optimizatori
import torch.optim as optim

# pentru tensorboard
from torch.utils.tensorboard import SummaryWriter


classes = ('burger', 'butter_naan', 'chai', 'chapati',
           'chole_bhature', 'dal_makhani', 'dhokla', 
           'fried_rice', 'idli', 'jalebi', 'kaathi_rolls',
           'kadai_paneer', 'kulfi', 'masala_dosa',
           'momos', 'paani_puri', 'pakode', 'pav_bhaji',
           'pizza', 'samosa')


################################## TRAIN FUNCTION ############################

def train_network(net, device, trainloader, val_generator, nr_epochs, SAVE_PATH):

    # pentru tensorboard, se instantiaza un writer:
    writer = SummaryWriter(log_dir='runs')
    # CRITERION = functia de loss.
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    #for epoch in tqdm.auto.tqdm(range(nr_epochs)):  # loop over the dataset multiple times
    for epoch in range(nr_epochs):

        running_loss = 0.0
        no_examples = 0
        #for i, data in tqdm.auto.tqdm(enumerate(trainloader)):
        for i, data in enumerate(tqdm(trainloader)):

            image, label = data
            #label = label[0]

            inputs = image.to(device).float()
            labels = label.to(device)
            #inputs, labels = data[0].to(device), data[1].to(device)

            # Se seteaza la ZERO gradientii optimkizarii (care erau nenuli de la iteratia precedenta)
            optimizer.zero_grad()
            #print(inputs.shape)
            outputs = net(inputs) # pur si simplu asa se face predictia: OUT = net(IN), unde net e instanta a Net

            loss = criterion(outputs, labels)
            
            loss.backward() # se calc. VALOAREA NUMERICA loss, aplicand functia de cost CRITERION, labelurilor
            optimizer.step() # se realizeaza efectiv backprop-ul, iterand prin TOTI TENSORII cu PARAMETRII            

            running_loss += loss.item()
            no_examples = i

            
        # pentru tensorboard
        writer.add_scalar("Loss/train", loss, epoch)

        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / no_examples:.3f}')

        #VALIDARE:
        current_acc = test_network(net, val_generator, device, classes, class_acc = False)
        writer.add_scalar("Acc/train", current_acc, epoch)

        # salvare model, daca e cel mai bun de pana acum:
        if epoch == 0:
                current_best_acc = current_acc
                torch.save(net.state_dict(), SAVE_PATH)
        elif current_acc <= current_best_acc:
                current_best_acc = current_acc
                torch.save(net.state_dict(), SAVE_PATH) 

               
    print('Finished Training')

    # se asigura ca toate datele sunt scrise bine pe disk
    writer.flush()
    # se inchide writer-ul. Nu se mai ppoate scrie inn el
    writer.close()

    return net # presupun ca e nevoie, ca sa am reteaua in maiin












