
# library
# standard library
import os
import time

import numpy as np

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
# torch.manual_seed(1)    # reproducible
from sklearn.metrics import r2_score
import torchvision.datasets as datasets
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
def accuracy(pd,gt):
    """Computes the precision@k for the specified values of k"""
    pd = np.array(pd)
    gt = np.array(gt)

    return np.mean(pd == gt)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(32*32*3,32*32*3),
            nn.ReLU(),
            nn.Linear(32*32*3,32*32*3),
            nn.ReLU(),
            nn.Linear(32 * 32 * 3, 32 * 32 * 3),
            nn.ReLU(),
            nn.Linear(32*32*3,10)
        )

    def forward(self, x):
        output = self.mlp(x)
        return output   # return x for visualization
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.261])
batch_size = 100
lr = 0.01
milestone = [50]
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
model = MLP().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum= 0.9,
                                weight_decay=1e-4)
EPOCHS = 100
loss_func = nn.CrossEntropyLoss()              # the target label is not one-hotted
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[50], last_epoch=- 1)
rmse = []
mae = []
rsquare = []

train_loss = []
val_loss = []
acc  =[]
times = 0
for epoch in range(EPOCHS):
    t = time.time()
    running_loss = 0
    model.train()
    for i,(input,target) in enumerate(train_loader):
        target = target.cuda()
        input = input.cuda()
        input = input.view(-1, 32*32*3)
        output = model(input)
        loss = criterion(output,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss.append(running_loss/50000)
    lr_scheduler.step()
    model.eval()
    pred = []
    gt = []
    valloss = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda().view(-1, 32*32*3)
            target_var = target.cuda()

            output = model(input_var)
            loss = criterion(output, target_var)

            _, preds = torch.max(output, dim=1)
            pred.append(preds.cpu().numpy())
            gt.append(target_var.cpu().numpy())
            valloss+= loss.item()
        val_loss.append(valloss/10000)

        acc.append(accuracy(pred, gt))

        print("epoch", epoch, ' acc:', accuracy(pred, gt))
    times += (time.time()-t)/EPOCHS
print(train_loss)
print(val_loss)
print(acc)
print(max(acc))
plt.plot([k for k in range(EPOCHS)], train_loss, label = 'train_loss',color = 'tab:blue')
plt.plot([k for k in range(EPOCHS)], val_loss,  label='val loss', color='tab:red')
plt.legend(loc = 'upper right')
plt.title('MLP')
plt.show()
plt.plot([k for k in range(EPOCHS)], acc, 'b.-')
plt.title('Accuracies of MLP for 100 epochs')
plt.show()





