import numpy as np
from scipy import signal
import pywt
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils
torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = torch.cuda.FloatTensor(X)
        self.Y = torch.cuda.LongTensor(Y)
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        if self.transform:
            x = self.transform(x)
        return x, y


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def load_data(data_path, subjects=[1,2,3,4,5,6,7,8,9], verbose=False):

    X_train_valid = np.load(data_path + "X_train_valid.npy")
    y_train_valid = np.load(data_path + "y_train_valid.npy") - 769
    person_train_valid = np.load(data_path + "person_train_valid.npy")

    X_test = np.load(data_path + "X_test.npy")
    y_test = np.load(data_path + "y_test.npy") - 769
    person_test = np.load(data_path + "person_test.npy")

    X_train_valid_subjects = np.empty(shape=[0, X_train_valid.shape[1], X_train_valid.shape[2]])
    y_train_valid_subjects = np.empty(shape=[0])
    X_test_subjects = np.empty(shape=[0, X_test.shape[1], X_test.shape[2]])
    y_test_subjects = np.empty(shape=[0])

    for s in subjects:

        # extract subject data
        X_train_valid_subject = X_train_valid[np.where(person_train_valid == s-1)[0], :, :]
        y_train_valid_subject = y_train_valid[np.where(person_train_valid == s-1)[0]]
        X_test_subject = X_test[np.where(person_test == s-1)[0], :, :]
        y_test_subject = y_test[np.where(person_test == s-1)[0]]

        # stack
        X_train_valid_subjects = np.concatenate((X_train_valid_subjects, X_train_valid_subject), axis=0)
        y_train_valid_subjects = np.concatenate((y_train_valid_subjects, y_train_valid_subject))
        X_test_subjects = np.concatenate((X_test_subjects, X_test_subject), axis=0)
        y_test_subjects = np.concatenate((y_test_subjects, y_test_subject))

    if verbose:
        print ('Training/Valid data shape: {}'.format(X_train_valid_subjects.shape))
        print ('Test data shape: {}'.format(X_test_subjects.shape))

    return X_train_valid_subjects, y_train_valid_subjects, X_test_subjects, y_test_subjects


def dataloader_setup(X_train_valid, y_train_valid, X_test, y_test, batch_size=32):

    perm = np.random.permutation(X_train_valid.shape[0])

    numTrain = int(0.8*X_train_valid.shape[0])
    numVal = X_train_valid.shape[0] - numTrain

    Xtrain = X_train_valid[perm[0:numTrain]][:, np.newaxis, :, :]
    ytrain = y_train_valid[perm[0:numTrain]]

    Xval = X_train_valid[perm[numTrain:]][:, np.newaxis, :, :]
    yval = y_train_valid[perm[numTrain:]]

    Xtest = X_test[:, np.newaxis, :, :]
    ytest = y_test

    # transformations
    transformations = transforms.Compose([
                        transforms.RandomErasing(p=.99,
                                              	scale=(.02,.08),
                                              	ratio=(.025,.026),
                                              	value=0),
                        AddGaussianNoise(mean=0., std=1.),
                      ])

    # load training dataset
    train_dataset = Dataset(Xtrain, ytrain, transform=transformations)
    train_loader = DataLoader(train_dataset, batch_size)

    # load validation dataset
    val_dataset = Dataset(Xval, yval)
    val_loader = DataLoader(val_dataset, numVal)

    # load test dataset
    test_dataset = Dataset(Xtest, ytest)
    test_loader = DataLoader(test_dataset, len(test_dataset))

    # package up
    data_loaders = [train_loader, val_loader, test_loader]

    return data_loaders


def filter_data(data, fs, order, lowcut, highcut):

    filtered_data = np.zeros_like(data)
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq

    for n in np.arange(data.shape[0]):
        single_instance = data[n, :, :]

        for channel in np.arange(single_instance.shape[0]):
            X = single_instance[channel, :]
            b, a = signal.butter(order, [low, high], btype='band')
            y = signal.lfilter(b, a, X)
            filtered_data[n, channel, :] = y

    return filtered_data


def smooth_data(data, ws):
    kern = signal.hanning(ws)[None, None, :]
    kern /= kern.sum()
    return signal.convolve(data, kern, mode='same')


def cwt_data(data, dt, wt_type, scales):
    wav = pywt.ContinuousWavelet(wt_type)
    coefs, freqs = pywt.cwt(data, scales, wav, sampling_period=0.01, method='fft')
    return np.array(coefs[0], dtype=np.float)

