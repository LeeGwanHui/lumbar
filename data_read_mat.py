## 필요한 패키지 등록
import os
import numpy as np
import matplotlib.pylab as plt
import scipy.io # matlab 파일 읽을 때
import glob #모름 glob는 파일들의 리스트를 뽑을 때 사용하는데, 파일의 경로명을 이용해서 입맛대로 요리할 수 있답니다.

## label 불러오기 및 train set 저장 구문
label_list = glob.glob('./train/label/*')

train_list = label_list[0:110]
dir_data = './datasets'
dir_save_train = os.path.join(dir_data, 'train')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

for i, filename in enumerate(train_list):
    mat_file = scipy.io.loadmat(filename)
    mat_file_value = mat_file['label_separated']
    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), mat_file_value)

## label 불러오기 및 val set 저장 구문
val_list= label_list[110:120]
dir_save_val = os.path.join(dir_data, 'val')

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

for i, filename in enumerate(val_list):
    mat_file = scipy.io.loadmat(filename)
    mat_file_value = mat_file['label_separated']
    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), mat_file_value)
