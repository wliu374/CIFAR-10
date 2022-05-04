
from __future__ import print_function

import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import time
import matplotlib.pyplot as plt
# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate

labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
# pick 2000 samples to speed up testing
train_data = torchvision.datasets.CIFAR10(root='./cifar-10/', train=True)
for i in range(5):
    plt.imshow(train_data.data[i])
    plt.title('label:'+labels[train_data.targets[i]])
    plt.axis('off')
    plt.show()
train_x = train_data.data/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
train_y = train_data.targets

# pick 2000 samples to speed up testing
test_data = torchvision.datasets.CIFAR10(root='./cifar-10/', train=False)

test_x = test_data.data/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.targets

# print(train_x.size(),train_y.size(),test_x.size(),test_y.size())
train_x = train_x.reshape((-1,32*32*3))
test_x = test_x.reshape((-1,32*32*3))
# train_x = train_x.view(-1,32*32*3)
# test_x = test_x.view(-1,32*32*3)
# print('train_x dimensions:',train_x.size())
# print('train_y dimensions:',train_y.size())
# print('test_x dimensions:',test_x.size())
# print('test_y dimensions:',test_y.size())

"""K-Nearest Neighbor Classification"""
def accuracy(pd,gt):
    """Computes the precision@k for the specified values of k"""
    pd = np.array(pd)
    gt = np.array(gt)

    return np.mean(pd == gt)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
# import imutils
# import cv2
from sklearn.model_selection import KFold

# load the MNIST digits dataset
# mnist = datasets.load_digits()
# print(len(np.array(mnist.data)[0]))

# Training and testing split,
# 75% for training and 25% for testing
# print(len(mnist))
# (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data), mnist.target, test_size=0.25, random_state=42)

# take 10% of the training data and use that for validation
# (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1, random_state=84)

trainData = np.array(train_x)
testData = np.array(test_x)
trainLabels = np.array(train_y)
testLabels = np.array(test_y)
valData = testData
valLabels = testLabels

# Checking sizes of each data split
print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))

# initialize the values of k for our k-Nearest Neighbor classifier along with the
# list of accuracies for each value of k
kVals = range(1, 30, 2)
accuracies = []
#
# # loop over kVals
t = time.time()
for k in range(1, 30, 2):
    # train the classifier with the current value of `k`
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainData, trainLabels)

    # evaluate the model and print the accuracies list
    score = model.score(valData, valLabels)
    print("k=%d, accuracy=%.4f%%" % (k, score * 100))
    accuracies.append(score)
print(time.time()-t)
plt.plot(range(1,30,2),accuracies,'b.-')
plt.savefig('results/score_vs_k',dpi = 300)
plt.show()
# largest accuracy
# np.argmax returns the indices of the maximum values along an axis
i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
    accuracies[i] * 100))

# Now that I know the best value of k, re-train the classifier
model = KNeighborsClassifier(n_neighbors=1)
model.fit(trainData, trainLabels)

# Predict labels for the test set
predictions = model.predict(testData)
print(accuracy(predictions,testLabels))
#
# # Evaluate performance of model for each of the digits
# print("EVALUATION ON TESTING DATA")
# print(classification_report(testLabels, predictions))
# print('accuracy:',accuracies[i],'r2',r2_score(testLabels,predictions))
# # some indices are classified correctly 100% of the time (precision = 1)
# # high accuracy (98%)
#
# # check predictions against images
# # loop over a few random digits
image = testData
j = 0
for i in np.random.randint(0, high=len(testLabels), size=(10)):
        # np.random.randint(low, high=None, size=None, dtype='l')
    prediction = model.predict(image)[i]
    if prediction != testLabels[i]:
        print(labels[testLabels[i]])
        plt.imshow(test_data.data[i])
        plt.axis('off')
        plt.title("predicted label: " + labels[prediction],fontsize = 16)
        plt.show()
        # plt.subplot(4,6,j+1)
        # plt.title("predicted label: "+str(prediction))
        # plt.imshow(image0)
        # plt.axis('off')




        # convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
        # then resize it to 32 x 32 pixels for better visualization

        #image0 = imutils.resize(image[0], width=32, inter=cv2.INTER_CUBIC)



    # show the prediction
    # print("I think that digit is: {}".format(prediction))
    # print('image0 is ',image0)
    # cv2.imshow("Image", image0)
    # cv2.waitKey(0) # press enter to view each one!
# plt.savefig("results/knn_score",dpi = 300)
# plt.show()

