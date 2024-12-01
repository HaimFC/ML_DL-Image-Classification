import numpy as np
import torch
import torch.nn as nn
def ID1():
    '''
        Personal ID of the first student.
    '''
    # Insert your ID here
    return 000000000

def ID2():
    '''
        Personal ID of the second student. Fill this only if you were allowed to submit in pairs, Otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000

class CNN(nn.Module):
    def __init__(self): # Do NOT change the signature of this function
        super(CNN, self).__init__()
        n = 8
        kernel_size = 3
        padding = (kernel_size - 1) // 2
        self.dsize = n
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=n,kernel_size=kernel_size,padding=padding)
        self.conv2 = nn.Conv2d(in_channels=n,out_channels=2*n,kernel_size=kernel_size,padding=padding)
        self.conv3 = nn.Conv2d(in_channels=2*n,out_channels=4*n,kernel_size=kernel_size,padding=padding)
        self.conv4 = nn.Conv2d(in_channels=4*n,out_channels=8*n,kernel_size=kernel_size,padding=padding)
        self.fc1 = nn.Linear(8*n * 28 * 14, 100)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(100, 2)
    
    def forward(self,inp):# Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor.
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        x = nn.functional.relu(self.conv1(inp))
        x = self.pool(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv4(x))
        x = self.pool(x)

        x = x.reshape(-1, 8 * self.dsize * 28 * 14)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

class CNNChannel(nn.Module):
    def __init__(self):# Do NOT change the signature of this function
        super(CNNChannel, self).__init__()
        n = 4
        self.dsize = n
        kernel_size = 5
        padding = (kernel_size - 1) // 2
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(in_channels=6,out_channels=n,kernel_size=kernel_size,padding=padding)
        self.conv2 = nn.Conv2d(in_channels=n,out_channels=2*n,kernel_size=kernel_size,padding=padding)
        self.conv3 = nn.Conv2d(in_channels=2*n,out_channels=4*n,kernel_size=kernel_size,padding=padding)
        self.conv4 = nn.Conv2d(in_channels=4*n,out_channels=8*n,kernel_size=kernel_size,padding=padding)
        self.fc1 = nn.Linear(8*n * 14 * 14, 100)
        self.fc2 = nn.Linear(100, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self,inp):# Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        top_half = inp[:, :, :224, :]
        bottom_half = inp[:, :, 224:, :]
        y = torch.cat((top_half, bottom_half), dim=1)
        y = self.pool(nn.functional.relu(self.conv1(y)))
        y = self.pool(nn.functional.relu(self.conv2(y)))
        y = self.pool(nn.functional.relu(self.conv3(y)))
        y = self.pool(nn.functional.relu(self.conv4(y)))
        y = y.reshape(-1, 8 * self.dsize * 14 * 14)
        y = nn.functional.relu(self.fc1(y))
        y = self.dropout(y)
        y = self.fc2(y)
        output = nn.functional.log_softmax(y, dim=1)

        return output