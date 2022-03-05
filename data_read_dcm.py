## 필요한 패키지 등록
import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import glob # glob는 파일들의 리스트를 뽑을 때 사용하는데, 파일의 경로명을 이용해서 입맛대로 요리할 수 있답니다.
import re

## 데이터 불러오기 및 train set 저장 구문 train : val =  110 : 10
data_list = glob.glob('./train/img/*')

train_list= data_list[0:110]
dir_data = './datasets'
dir_save_train = os.path.join(dir_data, 'train')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

for i,filename in enumerate(train_list):
    images = sitk.ReadImage(filename)
    images_array=sitk.GetArrayFromImage(images).astype('float32')
    img=np.squeeze(images_array)
    copy_img=img.copy()
    min=np.min(copy_img)
    max=np.max(copy_img)

    copy_img1=copy_img-np.min(copy_img)
    copy_img=copy_img1/np.max(copy_img1)
    copy_img*=2**8-1
    copy_img=copy_img.astype(np.uint8)

    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), copy_img)

## 데이터 불러오기 및 val set 저장 구문

val_list= data_list[110:120]
dir_save_val = os.path.join(dir_data, 'val')

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

for i,filename in enumerate(val_list):
    images = sitk.ReadImage(filename)
    images_array=sitk.GetArrayFromImage(images).astype('float32')
    img=np.squeeze(images_array)
    copy_img=img.copy()
    min=np.min(copy_img)
    max=np.max(copy_img)

    copy_img1=copy_img-np.min(copy_img)
    copy_img=copy_img1/np.max(copy_img1)
    copy_img*=2**8-1
    copy_img=copy_img.astype(np.uint8)

    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), copy_img)

## 데이터 불러오기 및 test set 저장 구문

# 데이터 저장하는 위치를 대입하면 된다.
data_list = glob.glob('./train/testdata_1/*')

test_list = data_list[:]
test_list.sort()
dir_data = './datasets'
# 데이터 저장하는 위치 변경해줘도 됨 'test'를 바꾸자
dir_save_test = os.path.join(dir_data, 'test')

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

for i, filename in enumerate(test_list):
    images = sitk.ReadImage(filename)
    images_array=sitk.GetArrayFromImage(images).astype('float32')
    img=np.squeeze(images_array)
    copy_img=img.copy()
    min=np.min(copy_img)
    max=np.max(copy_img)

    copy_img1=copy_img-np.min(copy_img)
    copy_img=copy_img1/np.max(copy_img1)
    copy_img*=2**8-1
    copy_img=copy_img.astype(np.uint8)

    np.save(os.path.join(dir_save_test,'input_%03d.npy' % i), copy_img)

## 확인해봐야 하는것 시험시간에 꼭 확인
data_list = glob.glob('./train/testdata_1/*')
test_list = data_list[:]
test_list.sort()