import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def train(model, optimizer, train_loader, epoch):
    
    # put the model into training mode
    model.train()
    
    # enumerate the train loader
    for i, data in enumerate(train_loader):
        
        # extract data from train loader (torch Variable)
        Xtrain = Variable(data[0])
        ytrain = Variable(data[1])
        
        # set optimizer gradient
        optimizer.zero_grad()
        
        # send input through model
        output = model(Xtrain)
        
        # calculate loss
        loss = F.cross_entropy(output, ytrain)
        
        # backprop
        loss.backward()
        
        # take a gradient step
        optimizer.step()
        
        # verbose
        if i % 10 == 0:
            print('Training Progress: \tEpoch {} [{}/{} ({:.2f}%)]\t\tLoss: {:.5f}'.format(
                epoch+1, i*len(Xtrain), len(train_loader.dataset), 100.*i/len(train_loader), loss.data))
    
    return model


def evaluate(model, data_loader, mode):
    
    # put the model into evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
    
    for i, data in enumerate(data_loader):
        
        # extract data from train loader (torch Variable)
        Xdata = Variable(data[0])
        ydata = Variable(data[1])
        
        # send input through model
        output = model(Xdata)
        
        # sum up batch loss
        test_loss += F.cross_entropy(output, ydata).data
        
        # find index of max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        
        # running sum of number of correct predictions
        correct += pred.eq(ydata.data.view_as(pred)).long().cpu().sum()

    # average test_loss
    test_loss /= len(data_loader.dataset)
    
    # verbose
    if mode == 'train':
        print('\tTrain loss: {:.5f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(data_loader.dataset), 100.*correct/len(data_loader.dataset)))
    
    elif mode == 'val':
        print('\tValidation loss: {:.5f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(data_loader.dataset), 100.*correct/len(data_loader.dataset)))
    
    elif mode == 'test':
        print('\tTest loss: {:.5f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), 100.*correct/len(data_loader.dataset)))
    
    else:
        pass
    
    return [test_loss, 1.*correct.item()/len(data_loader.dataset)]


def train_and_evaluate(model, optimizer, data_loaders, num_epochs=10):

    # unpackage data loaders
    train_loader = data_loaders[0]
    val_loader = data_loaders[1]
    test_loader = data_loaders[2]
    
    # initialize book keeping dictionary
    metrics = {}
    metrics['train'] = []
    metrics['val']  = []
    metrics['test'] = []

    # evaluate for each epoch and record
    for epoch in range(num_epochs):
        model = train(model, optimizer, train_loader, epoch)

        metrics['train'].append(evaluate(model, train_loader, mode='train'))
        metrics['val'].append(evaluate(model, val_loader, mode='val'))
        metrics['test'].append(evaluate(model, test_loader, mode='test'))

    metrics['train'] = np.array(metrics['train'])
    metrics['val']  = np.array(metrics['val'])
    metrics['test']  = np.array(metrics['test'])

    print('Best validation accuracy:')
    print(np.amax(metrics['val'][:, 1].data))

    print('Best test accuracy:')
    print(np.amax(metrics['test'][:, 1].data))

    plot(metrics, num_epochs)


def plot(metrics, num_epochs):

    fig, ax = plt.subplots(1, 2, figsize = (8, 4))
    
    ax[0].plot(range(num_epochs), metrics['train'][:, 0], range(num_epochs), metrics['val'][:, 0])
    ax[0].legend(['Train','Validation'])
    ax[0].set_title('Loss')

    ax[1].plot(range(num_epochs), metrics['train'][:, 1], range(num_epochs), metrics['val'][:, 1])
    ax[1].legend(['Train','Validation'])
    ax[1].set_title('Accuracy')

