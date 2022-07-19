

import torch

# pentru optimizatori
import torch.optim as optim
import torchvision

# pentru IMPORT-UL DATABASE si transformatele care i se aplica
from termios import PARODD
import torch
import torchvision # contine data loader-uri pentru seturi de date comune

# pentru reprezenatare
import matplotlib.pyplot as plt
import numpy as np

# pentru definirea neural network
import torch.nn as nn # tipuri de straturi
import torch.nn.functional as F # functii de activare

# pentru optimizatori
import torch.optim as optim

from retele import *


def test_network(net, test_generator, device, classes, class_acc = True):

    # se trece modelul in odul de evaluare
    net.eval()
    
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        #? ce semnifica axes (1, 2, 0)? 
        plt.show()
   # se incarca niste imagini
    dataiter = iter(test_generator)
    images, labels = dataiter.next()
    #? Dar de ce .next() mi le da pe primele 4??
    '''
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    SAVE_PATH = '/home/intern1/work/pytorch1/proiect_indian_food/modele_resize/model1.pth'
    # se incarca modelul salvat
    net = Net1() # se insdtantiaza un model chel
    net.load_state_dict(torch.load(SAVE_PATH)) # se incarca, in acea instanta, modelul salvat, cu .load_state_dict

    print(images.shape)
    # se face predictia:
    outputs = net(images[0:16,:,:,:].float())
    #print(outputs)
    #exit()
    '''
    ################ PE CIFRE ###################
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_generator:
            
            images, labels = data

            images = images.to(device).float()
            labels = labels.to(device)

            #print(images.shape, labels.shape)
            # calculate outputs by running images through the network
            #outputs = net(images)
            outputs = net(images[:,:,:,:].float())
            #print(outputs.shape)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if class_acc:
        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    else:
        print(f'Validation accuracy is: {100 * correct // total} %')
        return 100 * correct // total

    ## b) class accuracy

    if class_acc:
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # again no gradients needed
        with torch.no_grad():
            for data in test_generator:
                images, labels = data

                images = images.to(device).float()
                labels = labels.to(device)

                outputs = net(images[:,:,:,:].float())
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1


        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    
    # se trece din nou modelul in modul de antrenare
    net.train()















