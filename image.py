##
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from dataset import *
from torchvision import transforms, datasets
from model import UNet

## input 이미지 출력 확인
for i in range(20):
    plt.subplot(4, 5, i+1)
    data = np.load('./datasets/test/input_%03d.npy'%i)
    plt.imshow(data, cmap=plt.cm.bone)

## label 이미지 출력하기
for i in range(20):
    plt.subplot(4, 5, i+1)
    data = np.load('./datasets/data/label_%03d.npy'%(i+60))
    plt.imshow(data[...,0:6].sum(axis=2), cmap=plt.cm.bone)

## 이미지 resize 하는 법 -> 코랩에서 구현 함
# resize 크기 가져오기
data_dir = "./datasets"
transform = transforms.Compose([Normalization_test(mean=0.5, std=0.5), Resize_test(), ToTensor_test()])
dataset_test = Dataset_test(data_dir=os.path.join(data_dir, 'test'),transform=transform)
lst_data = os.listdir(os.path.join(data_dir, 'test'))
## 형태 파악
data = dataset_test.__getitem__(0)
input = data['input']
net=UNet()
output = net(input)
## shape 저장
shape_lst=[]
dir_data = './datasets'
dir_save_test = os.path.join(dir_data, 'test')

for i in range(len(lst_data)):
    k=np.load(os.path.join(dir_save_test,'input_%03d.npy'%i))
    revered_list = list(reversed(k.shape))
    b = tuple(revered_list)
    shape_lst.append(b)

## 테스트
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), Resize(), ToTensor()])
dataset_train = Dataset(data_dir=os.path.join('./datasets', 'train'), transform=transform)
##
data = dataset_train.__getitem__(110)
input = data['input']
label = data['label']
print(label.max())
print(label.min())
##
n= np.load('./datasets/test/input_000.npy')

plt.imshow(k[...,6:7].squeeze(), cmap='gray')

## 출력확인 겹쳐서 확인
import matplotlib.pyplot as plt

for i in range(20):
    data = dataset_train.__getitem__(i+100)
    input = data['input']
    label = data['label']
    plt.subplot(5, 4, i+1)
    plt.imshow(input.squeeze(), cmap=plt.cm.bone)
    plt.imshow(label[0:6,...].sum(axis=0), cmap='jet', alpha=0.5)  # interpolation='none'
