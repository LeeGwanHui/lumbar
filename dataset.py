## 라이브러리
import os
import numpy as np
import torch
from torchvision import transforms
from skimage.transform import resize
import cv2

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]
        lst_label.sort()
        lst_input.sort()
        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_input)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        #0~1사이로 normalize
        label = label
        input = input/255.0

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data

# test는 label이 없음
class Dataset_test(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_input = [f for f in lst_data if f.startswith('input')]
        lst_input.sort()
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_input)

    def __getitem__(self, index):
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        #0~1사이로 normalize
        input = input/255.0

        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input}

        if self.transform:
            data = self.transform(data)

        return data

## 트랜스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        input = input.transpose((2, 0, 1)).astype(np.float32)
        label = label.transpose((2, 0, 1)).astype(np.float32)

        data = {'input': torch.from_numpy(input), 'label': torch.from_numpy(label)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'input': input, 'label': label}

        return data

class Resize(object):
    def __init__(self, size=(512,512)):
        self.size = size
    def __call__(self, data):
        label, input = data['label'], data['input']

        #input= resize(input, self.size)
        #label= resize(label, self.size)
        input = cv2.resize(input, dsize=self.size, interpolation=cv2.INTER_CUBIC)[:, :, np.newaxis]
        label = cv2.resize(label, dsize=self.size, interpolation=cv2.INTER_CUBIC)

        data = {'input': input, 'label': label}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.7:
            label = np.fliplr(label)
            input = np.fliplr(input)

        elif np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data

class RandomRotation(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.7:
            label = np.rot90(label,k=1,axes=(0, 1))
            input = np.rot90(input,k=1)

        elif np.random.rand() > 0.5:
            label = np.rot90(label, k=3, axes=(0, 1))
            input = np.rot90(input, k=3)

        data = {'label': label, 'input': input}

        return data

## test 트랜스폼 구현하기 label 이 없음
class ToTensor_test(object):
    def __call__(self, data):
        input = data['input']

        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'input': torch.from_numpy(input)}

        return data

class Normalization_test(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        input = data['input']

        input = (input - self.mean) / self.std

        data = {'input': input}

        return data

class Resize_test(object):
    def __init__(self, size=(512,512)):
        self.size = size
    def __call__(self, data):
        input = data['input']

        input = cv2.resize(input, dsize=self.size, interpolation=cv2.INTER_CUBIC)[:, :, np.newaxis]

        data = {'input': input}

        return data

# ## 테스트
# transform = transforms.Compose([Normalization(mean=0.5, std=0.5), Resize(),RandomFlip(),RandomRotation(),ToTensor()])
# dataset_train = Dataset(data_dir=os.path.join('./datasets', 'train'), transform=transform)
# ##
# data = dataset_train.__getitem__(0)
# input = data['input']
# label = data['label']
#
# ## 출력확인 겹쳐서 확인
# import matplotlib.pyplot as plt
#
# for i in range(20):
#     data = dataset_train.__getitem__(i)
#     input = data['input']
#     label = data['label']
#     plt.subplot(5, 4, i+1)
#     plt.imshow(input.squeeze(), cmap=plt.cm.bone)
#     plt.imshow(label[0:6,...].sum(axis=0), cmap='jet', alpha=0.5)  # interpolation='none'
