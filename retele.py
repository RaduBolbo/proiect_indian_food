
import torch

# pentru definirea neural network
import torch.nn as nn # tipuri de straturi
import torch.nn.functional as F # functii de activare

'''

class Net1(nn.Module):
# asta nu ma prind de ce nu merge. Ii da Kill la testare.

    # in init se vor defini datele membre, adica TIPURILE DE STRATURI ce vor fi folosite, in metoda de forward
    def __init__(self):
        super().__init__()
        #? Daca nu trimit niciun argument, de ce mai apelez constructortul clasei de baza?
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.1) # oare mai merge crescut?
        #self.output_activation = nn.functional.softmax(10)
        self.conv1 = nn.Conv2d(3, 32, 8)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 6)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 4)
        self.batchnorm3 = nn.BatchNorm2d(128)
        
        self.fc3 = nn.Linear(460800, 20) # 10 neurponi la OUT, cate unul pentru fiecare clasa

    # metoda forward descrie forwardpropagation, folosid straturile definite ca date membre si intitializate in construtor (desi le hardcodez)
    def forward(self, x):
        #print('Shape of the input: ', x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.batchnorm1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.batchnorm2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.batchnorm3(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = nn.functional.softmax(self.fc3(x))
        x = self.fc3(x)
        return x

'''

class Net1(nn.Module):
    # Accc = 66% * NU CRED CA ASTA ERA

    # in init se vor defini datele membre, adica TIPURILE DE STRATURI ce vor fi folosite, in metoda de forward
    def __init__(self):
        super().__init__()
        #? Daca nu trimit niciun argument, de ce mai apelez constructortul clasei de baza?
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.1) # oare mai merge crescut?
        #self.output_activation = nn.functional.softmax(10)
        self.conv1 = nn.Conv2d(3, 16, 8)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 6)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4)
        self.batchnorm3 = nn.BatchNorm2d(64)
        
        self.fc3 = nn.Linear(230400, 20) # 10 neurponi la OUT, cate unul pentru fiecare clasa

    # metoda forward descrie forwardpropagation, folosid straturile definite ca date membre si intitializate in construtor (desi le hardcodez)
    def forward(self, x):
        #print('Shape of the input: ', x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.batchnorm1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.batchnorm2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.batchnorm3(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = nn.functional.softmax(self.fc3(x))
        x = self.fc3(x)
        return x

class Net2(nn.Module):
    # in init se vor defini datele membre, adica TIPURILE DE STRATURI ce vor fi folosite, in metoda de forward
    def __init__(self):
        super().__init__()
        #? Daca nu trimit niciun argument, de ce mai apelez constructortul clasei de baza?
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.1) # oare mai merge crescut?
        #self.output_activation = nn.functional.softmax(10)
        self.conv1 = nn.Conv2d(3, 32, 8)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 6)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 128, 4)
        self.batchnorm3 = nn.BatchNorm2d(128)
        
        self.fc3 = nn.Linear(460800, 20) # 10 neurponi la OUT, cate unul pentru fiecare clasa

    # metoda forward descrie forwardpropagation, folosid straturile definite ca date membre si intitializate in construtor (desi le hardcodez)
    def forward(self, x):
        #print('Shape of the input: ', x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.batchnorm1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.batchnorm2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.batchnorm3(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = nn.functional.softmax(self.fc3(x))
        x = self.fc3(x)
        return x


class Net3(nn.Module):
    # in init se vor defini datele membre, adica TIPURILE DE STRATURI ce vor fi folosite, in metoda de forward
    def __init__(self):
        super().__init__()
        #? Daca nu trimit niciun argument, de ce mai apelez constructortul clasei de baza?
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.40) # oare mai merge crescut?
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.20)
        self.dropout4 = nn.Dropout(0.10)
        #self.output_activation = nn.functional.softmax(10)
        self.conv1 = nn.Conv2d(3, 32, 24)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 14)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 8)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.batchnorm4 = nn.BatchNorm2d(128)
        
        self.fc3 = nn.Linear(21632, 20) # 10 neurponi la OUT, cate unul pentru fiecare clasa

    # metoda forward descrie forwardpropagation, folosid straturile definite ca date membre si intitializate in construtor (desi le hardcodez)
    def forward(self, x):
        #print('Shape of the input: ', x.shape)
        x = self.pool(F.relu(self.dropout1(self.conv1(x))))
        x = self.batchnorm1(x)
        x = self.pool(F.relu(self.dropout2(self.conv2(x))))
        x = self.batchnorm2(x)
        x = self.pool(F.relu(self.dropout3(self.conv3(x))))
        x = self.batchnorm3(x)
        x = self.pool(F.relu(self.dropout4(self.conv4(x))))
        x = self.batchnorm4(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = nn.functional.softmax(self.fc3(x))
        x = self.fc3(x)
        return x



class Net4(nn.Module):
    # in init se vor defini datele membre, adica TIPURILE DE STRATURI ce vor fi folosite, in metoda de forward
    def __init__(self):
        super().__init__()
        #? Daca nu trimit niciun argument, de ce mai apelez constructortul clasei de baza?
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.30) # oare mai merge crescut?
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.20)
        self.dropout4 = nn.Dropout(0.10)
        #self.output_activation = nn.functional.softmax(10)
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256, 3)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3)
        self.batchnorm4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 32, 3)
        self.batchnorm5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, 3, padding='same')
        self.batchnorm6 = nn.BatchNorm2d(32)

        
        self.fc3 = nn.Linear(288, 20) # 10 neurponi la OUT, cate unul pentru fiecare clasa

    # metoda forward descrie forwardpropagation, folosid straturile definite ca date membre si intitializate in construtor (desi le hardcodez)
    def forward(self, x):
        #print('Shape of the input: ', x.shape)
        x = self.pool(F.relu(self.dropout1(self.conv1(x))))
        x = self.batchnorm1(x)
        x = self.pool(F.relu(self.dropout2(self.conv2(x))))
        x = self.batchnorm2(x)
        x = self.pool(F.relu(self.dropout3(self.conv3(x))))
        x = self.batchnorm3(x)
        x = self.pool(F.relu(self.dropout4(self.conv4(x))))
        x = self.batchnorm4(x)

        x = self.pool(F.relu(self.conv5(x)))
        x = self.batchnorm5(x)
        x = self.pool(F.relu(self.conv6(x)))
        x = self.batchnorm6(x)
        x = F.relu(self.conv6(x))
        x = self.batchnorm6(x)
        x = F.relu(self.conv6(x))
        x = self.batchnorm6(x)

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = nn.functional.softmax(self.fc3(x))
        x = self.fc3(x)
        return x

class Net5(nn.Module):
    # in init se vor defini datele membre, adica TIPURILE DE STRATURI ce vor fi folosite, in metoda de forward
    def __init__(self):
        super().__init__()
        #? Daca nu trimit niciun argument, de ce mai apelez constructortul clasei de baza?
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.30) # oare mai merge crescut?
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.20)
        self.dropout4 = nn.Dropout(0.10)
        #self.output_activation = nn.functional.softmax(10)
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.batchnorm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, 3, padding = 'same')
        self.batchnorm4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, 3, padding = 'same')
        self.batchnorm5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 128, 3, padding = 'same')
        self.batchnorm6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 512, 3, padding = 'same')
        self.batchnorm7 = nn.BatchNorm2d(512)

        self.conv8 = nn.Conv2d(512, 512, 3, padding = 'same')
        self.batchnorm8 = nn.BatchNorm2d(512)

        self.conv9 = nn.Conv2d(512, 1024, 3, padding = 'same')
        self.batchnorm9 = nn.BatchNorm2d(1024)

        
        self.fc3 = nn.Linear(1024, 20) # 10 neurponi la OUT, cate unul pentru fiecare clasa

    # metoda forward descrie forwardpropagation, folosid straturile definite ca date membre si intitializate in construtor (desi le hardcodez)
    def forward(self, x):
        #print('Shape of the input: ', x.shape)
        x = self.pool(F.relu(self.dropout1(self.conv1(x))))
        x = self.batchnorm1(x)
        x = self.pool(F.relu(self.dropout2(self.conv2(x))))
        x = self.batchnorm2(x)
        x = self.pool(F.relu(self.dropout3(self.conv3(x))))
        x = self.batchnorm3(x)
        x = self.pool(F.relu(self.dropout3(self.conv4(x))))
        x = self.batchnorm4(x)
        x = self.pool(F.relu(self.dropout3(self.conv5(x))))
        x = self.batchnorm5(x)
        x = self.pool(F.relu(self.dropout3(self.conv6(x))))
        x = self.batchnorm6(x)
        x = self.pool(F.relu(self.dropout3(self.conv7(x))))
        x = self.batchnorm7(x)
        x = F.relu(self.dropout3(self.conv8(x)))
        x = self.batchnorm8(x)
        x = self.pool(F.relu(self.dropout3(self.conv9(x))))
        x = self.batchnorm9(x)

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = nn.functional.softmax(self.fc3(x))
        x = self.fc3(x)
        return x



class Net6(nn.Module):
    # in init se vor defini datele membre, adica TIPURILE DE STRATURI ce vor fi folosite, in metoda de forward
    # 312*312
    def __init__(self):
        super().__init__()
        #? Daca nu trimit niciun argument, de ce mai apelez constructortul clasei de baza?
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.35) # oare mai merge crescut?
        self.dropout2 = nn.Dropout(0.30)
        self.dropout3 = nn.Dropout(0.20)
        #self.output_activation = nn.functional.softmax(10)
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.batchnorm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, 3, padding = 'same')
        self.batchnorm4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, 3, padding = 'same')
        self.batchnorm5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 128, 3, padding = 'same')
        self.batchnorm6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 128, 3, padding = 'same')
        self.batchnorm7 = nn.BatchNorm2d(128)

        self.conv8 = nn.Conv2d(128, 512, 3, padding = 'same')
        self.batchnorm8 = nn.BatchNorm2d(512)

        self.conv9 = nn.Conv2d(512, 512, 3, padding = 'same')
        self.batchnorm9 = nn.BatchNorm2d(512)

        
        #self.fc3 = nn.Linear(4608, 20)
        self.fc3 = nn.Linear(2048, 20)
        

    # metoda forward descrie forwardpropagation, folosid straturile definite ca date membre si intitializate in construtor (desi le hardcodez)
    def forward(self, x):
        #print('Shape of the input: ', x.shape)
        x = self.pool(F.relu(self.dropout1(self.conv1(x))))
        x = self.batchnorm1(x)
        x = self.pool(F.relu(self.dropout2(self.conv2(x))))
        x = self.batchnorm2(x)
        x = self.pool(F.relu(self.dropout3(self.conv3(x))))
        x = self.batchnorm3(x)
        x = self.pool(F.relu(self.dropout3(self.conv4(x))))
        x = self.batchnorm4(x)
        x = self.pool(F.relu(self.dropout3(self.conv5(x))))
        x = self.batchnorm5(x)
        x = self.pool(F.relu(self.dropout3(self.conv6(x))))
        x = self.batchnorm6(x)
        x = self.pool(F.relu(self.conv7(x)))
        x = self.batchnorm7(x)
        x = F.relu(self.conv8(x))
        x = self.batchnorm8(x)
        x = F.relu(self.conv9(x))
        x = self.batchnorm9(x)

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = nn.functional.softmax(self.fc3(x))
        x = self.fc3(x)
        return x



class Net7(nn.Module):
    # in init se vor defini datele membre, adica TIPURILE DE STRATURI ce vor fi folosite, in metoda de forward
    def __init__(self):
        super().__init__()
        #? Daca nu trimit niciun argument, de ce mai apelez constructortul clasei de baza?
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.35) # oare mai merge crescut?
        self.dropout2 = nn.Dropout(0.30)
        self.dropout3 = nn.Dropout(0.20)
        #self.output_activation = nn.functional.softmax(10)
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.batchnorm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 3, padding = 'same')
        self.batchnorm4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 128, 3, padding = 'same')
        self.batchnorm5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 128, 3, padding = 'same')
        self.batchnorm6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 128, 3, padding = 'same')
        self.batchnorm7 = nn.BatchNorm2d(128)

        self.conv8 = nn.Conv2d(128, 128, 3, padding = 'same')
        self.batchnorm8 = nn.BatchNorm2d(128)

        self.conv9 = nn.Conv2d(128, 128, 3, padding = 'same')
        self.batchnorm9 = nn.BatchNorm2d(128)

        self.conv10 = nn.Conv2d(128, 256, 3, padding = 'same')
        self.batchnorm10 = nn.BatchNorm2d(256)

        self.conv11 = nn.Conv2d(256, 256, 3, padding = 'same')
        self.batchnorm11 = nn.BatchNorm2d(256)

        self.conv12 = nn.Conv2d(256, 256, 3, padding = 'same')
        self.batchnorm12 = nn.BatchNorm2d(256)
        '''
        self.conv13 = nn.Conv2d(128, 256, 3, padding = 'same')
        self.batchnorm13 = nn.BatchNorm2d(256)

        self.conv14 = nn.Conv2d(256, 256, 3, padding = 'same')
        self.batchnorm14 = nn.BatchNorm2d(256)
        '''
        
        self.fc3 = nn.Linear(2304, 20) # 10 neurponi la OUT, cate unul pentru fiecare clasa

    # metoda forward descrie forwardpropagation, folosid straturile definite ca date membre si intitializate in construtor (desi le hardcodez)
    def forward(self, x):
        #print('Shape of the input: ', x.shape)
        x = self.pool(F.relu(self.dropout1(self.conv1(x))))
        x = self.batchnorm1(x)
        x = self.pool(F.relu(self.dropout2(self.conv2(x))))
        x = self.batchnorm2(x)
        x = self.pool(F.relu(self.dropout3(self.conv3(x))))
        x = self.batchnorm3(x)
        x = self.pool(F.relu(self.dropout3(self.conv4(x))))
        x = self.batchnorm4(x)
        x = self.pool(F.relu(self.dropout3(self.conv5(x))))
        x = self.batchnorm5(x)
        x = self.pool(F.relu(self.dropout3(self.conv6(x))))
        x = self.batchnorm6(x)
        x = self.pool(F.relu(self.conv7(x)))
        x = self.batchnorm7(x)
        x = F.relu(self.conv8(x))
        x = self.batchnorm8(x)
        x = F.relu(self.conv9(x))
        x = self.batchnorm9(x)
        x = F.relu(self.conv10(x))
        x = self.batchnorm10(x)
        x = F.relu(self.conv11(x))
        x = self.batchnorm11(x)
        x = F.relu(self.conv12(x))
        x = self.batchnorm12(x)
        '''
        x = F.relu(self.conv13(x))
        x = self.batchnorm13(x)
        x = F.relu(self.conv14(x))
        x = self.batchnorm14(x)
        '''

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = nn.functional.softmax(self.fc3(x))
        x = self.fc3(x)
        return x



class Net8(nn.Module):
    # se incearca o ARHITECTURA MAI SIMPLA
    # # 412*412 

    # in init se vor defini datele membre, adica TIPURILE DE STRATURI ce vor fi folosite, in metoda de forward
    def __init__(self):
        super().__init__()
        #? Daca nu trimit niciun argument, de ce mai apelez constructortul clasei de baza?
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.1) # oare mai merge crescut?
        #self.output_activation = nn.functional.softmax(10)
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 128, 3)
        self.batchnorm3 = nn.BatchNorm2d(128)
        
        self.fc3 = nn.Linear(307328, 20) # 10 neurponi la OUT, cate unul pentru fiecare clasa

    # metoda forward descrie forwardpropagation, folosid straturile definite ca date membre si intitializate in construtor (desi le hardcodez)
    def forward(self, x):
        #print('Shape of the input: ', x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.batchnorm1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.batchnorm2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.batchnorm3(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = nn.functional.softmax(self.fc3(x))
        x = self.fc3(x)
        return x



class Net9(nn.Module):
    # in init se vor defini datele membre, adica TIPURILE DE STRATURI ce vor fi folosite, in metoda de forward
    # 512*512
    def __init__(self):
        super().__init__()
        #? Daca nu trimit niciun argument, de ce mai apelez constructortul clasei de baza?
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.30)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.15)

        self.conv1 = nn.Conv2d(3, 32, 5) # $$$$$$$$$$$$$$$$$$$ filtru de 3 ?
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.batchnorm3 = nn.BatchNorm2d(64)

        # $$$$$$$$$$$$$$$$$$$$$$$ crescut dimensiunea iamginii

        self.conv4 = nn.Conv2d(64, 64, 3, padding = 'same')
        self.batchnorm4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, 3, padding = 'same')
        self.batchnorm5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 128, 3, padding = 'same')
        self.batchnorm6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 128, 3, padding = 'same')
        self.batchnorm7 = nn.BatchNorm2d(128)

        self.conv8 = nn.Conv2d(128, 256, 3, padding = 'same') # $$$$$$$ scazut nr filtre?
        self.batchnorm8 = nn.BatchNorm2d(256)

        self.conv9 = nn.Conv2d(256, 256, 3, padding = 'same') # $$$$$$$ scazut nr filtre?
        self.batchnorm9 = nn.BatchNorm2d(256)
        
        self.fc3 = nn.Linear(2304, 20)
        

    # metoda forward descrie forwardpropagation, folosid straturile definite ca date membre si intitializate in construtor (desi le hardcodez)
    def forward(self, x):
        #print('Shape of the input: ', x.shape)
        x = self.pool(F.relu(self.dropout1(self.conv1(x))))
        x = self.batchnorm1(x)
        x = self.pool(F.relu(self.dropout2(self.conv2(x))))
        x = self.batchnorm2(x)
        x = self.pool(F.relu(self.dropout3(self.conv3(x))))
        x = self.batchnorm3(x)
        x = self.pool(F.relu(self.dropout3(self.conv4(x))))
        x = self.batchnorm4(x)
        x = self.pool(F.relu(self.dropout3(self.conv5(x))))
        x = self.batchnorm5(x)
        x = self.pool(F.relu(self.dropout3(self.conv6(x))))
        x = self.batchnorm6(x)
        x = self.pool(F.relu(self.conv7(x)))
        x = self.batchnorm7(x)
        x = F.relu(self.conv8(x))
        x = self.batchnorm8(x)
        x = F.relu(self.conv9(x))
        x = self.batchnorm9(x)

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = nn.functional.softmax(self.fc3(x))
        x = self.fc3(x)
        return x



class Net10(nn.Module):
    # in init se vor defini datele membre, adica TIPURILE DE STRATURI ce vor fi folosite, in metoda de forward
    # 412*412
    def __init__(self):
        super().__init__()
        #? Daca nu trimit niciun argument, de ce mai apelez constructortul clasei de baza?
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.45)
        self.dropout2 = nn.Dropout(0.35)
        self.dropout3 = nn.Dropout(0.15)

        self.conv1 = nn.Conv2d(3, 32, 3) # $$$$$$$$$$$$$$$$$$$ filtru de 3 ?
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.batchnorm3 = nn.BatchNorm2d(64)

        # $$$$$$$$$$$$$$$$$$$$$$$ crescut dimensiunea iamginii

        self.conv4 = nn.Conv2d(64, 64, 3, padding = 'same')
        self.batchnorm4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, 3, padding = 'same')
        self.batchnorm5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 128, 3, padding = 'same')
        self.batchnorm6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 128, 3, padding = 'same')
        self.batchnorm7 = nn.BatchNorm2d(128)

        self.conv8 = nn.Conv2d(128, 256, 3, padding = 'same') # $$$$$$$ scazut nr filtre?
        self.batchnorm8 = nn.BatchNorm2d(256)

        self.conv9 = nn.Conv2d(256, 256, 3, padding = 'same') # $$$$$$$ scazut nr filtre?
        self.batchnorm9 = nn.BatchNorm2d(256)
        
        self.fc3 = nn.Linear(2304, 20)
        

    # metoda forward descrie forwardpropagation, folosid straturile definite ca date membre si intitializate in construtor (desi le hardcodez)
    def forward(self, x):
        #print('Shape of the input: ', x.shape)
        x = self.pool(F.relu(self.dropout1(self.conv1(x))))
        x = self.batchnorm1(x)
        x = self.pool(F.relu(self.dropout2(self.conv2(x))))
        x = self.batchnorm2(x)
        x = self.pool(F.relu(self.dropout3(self.conv3(x))))
        x = self.batchnorm3(x)
        x = self.pool(F.relu(self.dropout3(self.conv4(x))))
        x = self.batchnorm4(x)
        x = self.pool(F.relu(self.dropout3(self.conv5(x))))
        x = self.batchnorm5(x)
        x = F.relu(self.dropout3(self.conv6(x)))
        x = self.batchnorm6(x)
        x = F.relu(self.conv7(x))
        x = self.batchnorm7(x)
        x = self.pool(F.relu(self.conv8(x)))
        x = self.batchnorm8(x)
        x = self.pool(F.relu(self.conv9(x)))
        x = self.batchnorm9(x)

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = nn.functional.softmax(self.fc3(x))
        x = self.fc3(x)
        return x






























































































