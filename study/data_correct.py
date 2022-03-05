##
import numpy as np
import matplotlib.pyplot as plt
import os

data_org = np.load('./datasets/train/label_%03d.npy' % 59)
data = data_org[..., 6: 7]
data[data==-254]=0

data_org[...,6:7] = data

##
dir_data = './datasets'
dir_save_train = os.path.join(dir_data, 'train')

np.save(os.path.join(dir_save_train, 'label_%03d.npy' % 59), data_org)

##
plt.imshow(data_org[...,0:1].squeeze(), cmap=plt.cm.bone)
data_org.min()
##
data_org = np.load('./datasets/train/label_%03d.npy' % 3)

##
data_org = np.load('./datasets/train/input_%03d.npy' % 2)
plt.imshow(data_org.squeeze(), cmap=plt.cm.bone)

