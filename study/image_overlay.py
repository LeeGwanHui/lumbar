##
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io

# Both images are loaded from a dicom. Both are numpy arrays of (512,512)
Image1 = np.load('../datasets/train/input_000.npy')
mat_file_name = "../train/label/001.mat"
mat_file = scipy.io.loadmat(mat_file_name)
Image2 = mat_file['label_separated']

img_label=np.ones((3268,3268))
for i in range(6):
    img_label = img_label + Image2[:, :, i]

## Plot images
plt.imshow(Image1,cmap=plt.cm.bone)
plt.imshow(img_label, cmap='jet', alpha=0.1) # interpolation='none'
