## 필요한 패키지 import
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import cv2

##
filename = './train/img/005.dcm'
images = sitk.ReadImage(filename)

print("Width: ",images.GetWidth())
print("Heigh: ",images.GetHeight())
print("Depth: ",images.GetDepth())
print("Dimension: ",images.GetDimension())
print("Pixel ID:",images.GetPixelIDValue())
print("Pixel ID Type: ",images.GetPixelIDTypeAsString())

images_array=sitk.GetArrayFromImage(images).astype('float32')
img=np.squeeze(images_array)
copy_img=img.copy()
min=np.min(copy_img)
max=np.max(copy_img)

copy_img1=copy_img-np.min(copy_img)
copy_img=copy_img1/np.max(copy_img1)
copy_img*=2**8-1
copy_img=copy_img.astype(np.uint8)

## gray -> RGB 지금 프로젝트에 필요없음
copy_img=np.expand_dims(copy_img,axis=-1)
copy_img=cv2.cvtColor(copy_img,cv2.COLOR_GRAY2RGB)

## 출력 확인
plt.subplot(121)
plt.imshow(copy_img)
plt.subplot(122)
plt.imshow(img,cmap=plt.cm.bone)

## 파일명으로 출력 확인
data = np.load('./datasets/train/input_000.npy')
plt.imshow(data,cmap=plt.cm.bone)
plt.title('input')