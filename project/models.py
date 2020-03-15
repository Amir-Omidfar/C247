import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

DROPOUT = 0.4

class Permute(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 1, 3)

'''
CNN
'''
class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn = nn.Sequential(

            # Input: N x 1 x 22 x 1000

            ### Conv-Pool Block 1
            # Convolution (temporal)
            nn.Conv2d(1, 25, kernel_size=(1, 10), stride=1, padding=0),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(25, 25, kernel_size=(3, 3), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(25, 25, kernel_size=(3, 3), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(25, 25, kernel_size=(18, 1), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.2, affine=True),
            
            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            
            # Dropout
            nn.Dropout(p=DROPOUT),


            ### Conv-Pool Block 2
            # Convolution
            nn.Conv2d(1, 50, kernel_size=(25, 10), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=50, eps=1e-05, momentum=0.2, affine=True),

            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            # Dropout
            nn.Dropout(p=DROPOUT),


            ### Conv-Pool Block 3
            # Convolution
            nn.Conv2d(1, 100, kernel_size=(50, 10), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=100, eps=1e-05, momentum=0.2, affine=True),

            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            # Dropout
            nn.Dropout(p=DROPOUT),


            ### Conv-Pool Block 4
            # Convolution
            nn.Conv2d(1, 200, kernel_size=(100, 10), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=200, eps=1e-05, momentum=0.2, affine=True),
            
            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

        )
        
        self.fc = nn.Sequential(
            nn.Linear(200, 54),
            nn.BatchNorm1d(num_features=54, eps=1e-05, momentum=0.2, affine=True),
            nn.ReLU(inplace = True),
            nn.Dropout(p=DROPOUT),
            nn.Linear(54, 44),
            nn.BatchNorm1d(num_features=44, eps=1e-05, momentum=0.2, affine=True),
            nn.ReLU(inplace = True),
            nn.Linear(44, 4)
        )

    def forward(self, x):
        
        # CNN
        x = self.cnn(x)

        N, C, H, W = x.size()
        x = x.view(N, H, W).permute(0, 2, 1)
        
        # Fully Connected Layer
        out = self.fc(x[:, -1, :])

        return out


'''
LSTM
'''
class LSTM(nn.Module):
    
    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(22, 64, 3, batch_first=True, dropout=DROPOUT)

        self.fc = nn.Sequential(
            nn.Linear(64, 54),
            nn.BatchNorm1d(num_features=54, eps=1e-05, momentum=0.2, affine=True),
            nn.ReLU(inplace = True),
            nn.Dropout(p=DROPOUT),
            nn.Linear(54, 44),
            nn.BatchNorm1d(num_features=44, eps=1e-05, momentum=0.2, affine=True),
            nn.ReLU(inplace = True),
            nn.Linear(44, 4)
        )
    
    def forward(self, x, h=None):

        # LSTM
        N, C, H, W = x.size()
        x = x.view(N, H, W).permute(0, 2, 1)
        out, _ = self.lstm(x)

        # Fully Connected Layer
        out = self.fc(out[:, -1, :])

        return out


'''
GRU
'''
class GRU(nn.Module):
    
    def __init__(self):
        super(GRU, self).__init__()

        self.gru = nn.GRU(22, 64, 3, batch_first=True, dropout=DROPOUT)

        self.fc = nn.Sequential(
            nn.Linear(64, 54),
            nn.BatchNorm1d(num_features=54, eps=1e-05, momentum=0.2, affine=True),
            nn.ReLU(inplace = True),
            nn.Dropout(p=DROPOUT),
            nn.Linear(54, 44),
            nn.BatchNorm1d(num_features=44, eps=1e-05, momentum=0.2, affine=True),
            nn.ReLU(inplace = True),
            nn.Linear(44, 4)
        )
    
    def forward(self, x, h=None):

        # GRU
        N, C, H, W = x.size()
        x = x.view(N, H, W).permute(0, 2, 1)
        out, _ = self.gru(x)

        # Fully Connected Layer
        out = self.fc(out[:, -1, :])

        return out


'''
CNN + LSTM
'''
class CNN_LSTM(nn.Module):
    
    def __init__(self):
        super(CNN_LSTM, self).__init__()

        self.cnn = nn.Sequential(

            # Input: N x 1 x 22 x 1000

            ### Conv-Pool Block 1
            # Convolution (temporal)
            nn.Conv2d(1, 25, kernel_size=(1, 10), stride=1, padding=0),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(25, 25, kernel_size=(3, 3), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(25, 25, kernel_size=(3, 3), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(25, 25, kernel_size=(18, 1), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.2, affine=True),
            
            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            
            # Dropout
            nn.Dropout(p=DROPOUT),


            ### Conv-Pool Block 2
            # Convolution
            nn.Conv2d(1, 50, kernel_size=(25, 10), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=50, eps=1e-05, momentum=0.2, affine=True),

            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            # Dropout
            nn.Dropout(p=DROPOUT),


            ### Conv-Pool Block 3
            # Convolution
            nn.Conv2d(1, 100, kernel_size=(50, 10), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=100, eps=1e-05, momentum=0.2, affine=True),

            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            # Dropout
            nn.Dropout(p=DROPOUT),


            ### Conv-Pool Block 4
            # Convolution
            nn.Conv2d(1, 200, kernel_size=(100, 10), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=200, eps=1e-05, momentum=0.2, affine=True),
            
            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

        )
        
        self.lstm = nn.LSTM(7, 64, 3, batch_first=True, dropout=DROPOUT)
        
        self.fc = nn.Sequential(
            nn.Linear(64, 4),
        )

    
    def forward(self, x):

        # CNN
        x = self.cnn(x)

        # LSTM
        N, C, H, W = x.size()
        x = x.view(N, H, W).permute(0, 1, 2)
        out, _ = self.lstm(x)

        # Fully Connected Layer
        out = self.fc(out[:, -1, :])

        return out


'''
CNN + GRU
'''
class CNN_GRU(nn.Module):
    
    def __init__(self):
        super(CNN_GRU, self).__init__()

        self.cnn = nn.Sequential(

            # Input: N x 1 x 22 x 1000

            ### Conv-Pool Block 1
            # Convolution (temporal)
            nn.Conv2d(1, 25, kernel_size=(1, 10), stride=1, padding=0),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(25, 25, kernel_size=(3, 3), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(25, 25, kernel_size=(3, 3), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(25, 25, kernel_size=(18, 1), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.2, affine=True),
            
            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            
            # Dropout
            nn.Dropout(p=DROPOUT),


            ### Conv-Pool Block 2
            # Convolution
            nn.Conv2d(1, 50, kernel_size=(25, 10), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=50, eps=1e-05, momentum=0.2, affine=True),

            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            # Dropout
            nn.Dropout(p=DROPOUT),


            ### Conv-Pool Block 3
            # Convolution
            nn.Conv2d(1, 100, kernel_size=(50, 10), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=100, eps=1e-05, momentum=0.2, affine=True),

            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            # Dropout
            nn.Dropout(p=DROPOUT),


            ### Conv-Pool Block 4
            # Convolution
            nn.Conv2d(1, 200, kernel_size=(100, 10), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=200, eps=1e-05, momentum=0.2, affine=True),
            
            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

        )
        
        self.gru = nn.GRU(7, 64, 3, batch_first=True, dropout=DROPOUT)
        
        self.fc = nn.Sequential(
            nn.Linear(64, 4),
        )

    
    def forward(self, x):

        # CNN
        x = self.cnn(x)

        # GRU
        N, C, H, W = x.size()
        x = x.view(N, H, W).permute(0, 1, 2)
        out, _ = self.gru(x)

        # Fully Connected Layer
        out = self.fc(out[:, -1, :])

        return out

    