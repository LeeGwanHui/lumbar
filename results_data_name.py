## result 결과 파일 자동 rename
import os
import glob
import re

## test set 파일명 불러오기
test_original_list = glob.glob('./train/testdata_1/*')
test_original_list.sort()
new_name = []
for i in range(len(test_original_list)):
    new_name.append(re.sub(r'[^0-9]', '', os.path.basename(test_original_list[i])))
new_name.sort()
##
path = './results/numpy'
files = glob.glob(path + '/*')
files.sort()

for i, f in enumerate(files):
    os.rename(f, os.path.join(path, new_name[i])+'.npy')


##
path = './results/numpy'
files = glob.glob(path + '/*')
files.sort()