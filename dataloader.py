
import cv2
import torch
import os
import numpy as np
import imutils
import random

classes = ('burger', 'butter_naan', 'chai', 'chapati',
           'chole_bhature', 'dal_makhani', 'dhokla', 
           'fried_rice', 'idli', 'jalebi', 'kaathi_rolls',
           'kadai_paneer', 'kulfi', 'masala_dosa',
           'momos', 'paani_puri', 'pakode', 'pav_bhaji',
           'pizza', 'samosa')


def resize (img):
    img = cv2.resize (img, (512, 512), interpolation = cv2.INTER_AREA)
    return img
'''
def one_hot_encoding(label_list):
    unique, inverse = np.unique(label_list, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot
'''
def one_hot_encoding(label_list):
    unique, inverse = np.unique(label_list, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return inverse

class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_path, color_base, zoom = False, rotate = False):
        names=[]
        labels=[]
        for label_name in os.listdir(img_path):
            for img_name in os.listdir(img_path + '/' + label_name):
                names.append(img_name)
                labels.append(label_name)
        
        # se transforma labels, din lista de str, in pne hut


        self.names = names
        self.labels = labels
        self.encoded_labels = one_hot_encoding(self.labels)
        self.img_path = img_path
        self.color_base = color_base
        self.rotate = rotate
        self.zoom = zoom


    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        ID = self.names[index]
        #x = torch.load(self.img_path + self.labels[index] + '/' +ID)
        x = cv2.imread(self.img_path + self.labels[index] + '/' +ID)
        
        # Resize:
        x = resize(x)

        # convert to hsv
        if self.color_base == 'hsv':
            x = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
            x[:,:,0], x[:,:,1], x[:,:,2] = x[:,:,0]/180, x[:,:,1]/255, x[:,:,2]/255
        else:
            x = x/255
        '''
        x = np.swapaxes(x, 2, 0)
        x = np.swapaxes(x, 2, 1)
        '''
        # rotate:
        #print('x before:')
        #print(x.shape)
        if self.rotate:
            # se genereaza unghiul de rotatie, dinbtr-oi distributie exponentiala.
            rot_angle = int(np.ceil(10*np.random.exponential(scale = 1)))
            # se plafoneaza unghiul, in cazutile extreme
            if rot_angle>50:
                rot_angle = 50
            # se roteste imagionea
            #x = imutils.rotate(x, rot_angle)
            x = imutils.rotate_bound(x, rot_angle)
        x = resize(x)

        #cv2.imshow('ai', x)
        #cv2.waitKey()
        x = np.swapaxes(x, 2, 0)
        x = np.swapaxes(x, 2, 1)
        
        #print('x after:')
        #print(x.shape)
        #exit()

        

        y = classes.index(str(self.labels[index]))
        

        #y = index
        return x,y
    
    def afisare(self):
        for index in range(len(self.names)):
            ID = self.names[index]
            print(self.img_path + self.labels[index] + '/' +ID)










