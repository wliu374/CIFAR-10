import matplotlib.pyplot as plt
import torchvision.models as models
from sklearn.metrics import r2_score
import argparse
import os
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
import time
from VGG import *
def compute_mae(pd, gt):
    pd, gt = np.array(pd), np.array(gt)
    diff = pd - gt
    mae = np.mean(np.abs(diff))
    return mae

def compute_rmse(pd, gt):
    pd, gt = np.array(pd), np.array(gt)
    diff = pd - gt
    mse = np.sqrt(np.mean((diff ** 2)))
    return mse

def rsquared(pd, gt):
    """ Return R^2 where x and y are array-like."""
    return r2_score(gt,pd)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.261])
batch_size = 100
lr = 0.01
milestone = [20,50]
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]), download=True),
    batch_size=batch_size, shuffle=True,)

val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),shuffle=False)

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda()

model = pretrainedVGG().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum= 0.9,
                                weight_decay=1e-4)
EPOCHS = 100
def accuracy(pd,gt):
    """Computes the precision@k for the specified values of k"""
    pd = np.array(pd)
    gt = np.array(gt)

    return np.mean(pd == gt)

train_loss = []
val_loss = []
acc = []
tra_acc = []
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[30,60], last_epoch=- 1)
times = 0
for epoch in range(EPOCHS):
    running_loss = 0
    model.train()
    err_sample = 0
    t = time.time()
    for i,(input,target) in enumerate(train_loader):
        target = target.cuda()
        input = input.cuda()

        output = model(input)
        loss = criterion(output,target)
        _, pred = torch.max(output, dim=1)
        err_sample += accuracy(pred.cpu().numpy(),target.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    tra_acc.append(err_sample/500)
    train_loss.append(running_loss/500)
    lr_scheduler.step()
    model.eval()
    pred = []
    gt = []
    valloss = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            output = model(input_var)
            loss = criterion(output, target_var)

            _, preds = torch.max(output, dim=1)
            pred.append(preds.cpu().numpy())
            gt.append(target_var.cpu().numpy())
            valloss+= loss.item()
        val_loss.append(valloss/10000)

        acc.append(accuracy(pred,gt))

        print("epoch",epoch,' acc:',accuracy(pred,gt))
    times += (time.time() - t)/EPOCHS
print(train_loss)
print(val_loss)
print(acc)
print(max(acc))
plt.plot([k for k in range(EPOCHS)], train_loss, label = 'train_loss',color = 'tab:blue')
plt.plot([k for k in range(EPOCHS)], val_loss,  label='val loss', color='tab:red')
plt.legend(loc = 'upper right')
plt.title('VGG19')
plt.show()

plt.plot([k for k in range(EPOCHS)], acc, 'r.-',label = 'val_accuracy')
plt.plot([k for k in range(EPOCHS)], tra_acc, 'b.-',label ='tra_accuracy')
plt.legend(loc = 'lower right')
plt.title('Accuracies of VGG19 for 100 epoohs')
plt.show()
print('time',times)
