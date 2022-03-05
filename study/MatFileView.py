##
import numpy as np
import matplotlib.pylab as plt
import scipy.io

mat_file_name = "./train/label/001.mat"
mat_file = scipy.io.loadmat(mat_file_name)
print(type(mat_file))

##
for i in mat_file:
    print(i)
mat_file_value = mat_file['label_separated']
print("size :", mat_file_value.shape)

## 이미지 출력하기
img_label=np.zeros((3268,3268)).astype(np.uint8)
for i in range(6):
    img_label = img_label + mat_file_value[:, :, i]
plt.imshow(img_label)
