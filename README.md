# lumbar
이 프로젝트는 대학교의 황도식 교수님 기초인공지능 수업의 최종 프로젝트 lumbar + spine segmentation을 진행한 파일입니다. 
이 프로젝트는 처음부터 데이터만 주어지고 모든 것을 새로 짰어야 하므로 많은 참고 문헌이 있는데 다 기억하지는 못하고 가장 도움 받은 곳은 아래 유튜브 링크입니다.

https://www.youtube.com/watch?v=kVaBDpwgsGg&list=PLqtXapA2WDqbE6ghoiEJIrmEnndQ7ouys

이 프로젝트의 핵심은 


내 파일 매뉴얼  
0. 파일 압축해제 후에 train중 testdata_1 파일 넣기 (폴더명 바꿀라면  data_read_dcm.py코드에서 파일명만 변경하면됨)
-> datasets 에 test data 삭제후 
1.  dcm - >  npy        :   data_read_dcm.py 59번째 줄 실행 84번줄에서 순서대로인지 확인

2. colab run 파일로 가서 test 돌리기( 제일 아래 mark되있는곳 돌리기 위에꺼는 256x256 모델이고 제일 아래 모델은 512x512이다.)

3. results_data_name.py 돌리기 ( 이름을 원래 네임대로 바꾸어주는 역할을 함)

# 파일 설명
 data_read_dcm.py -> dcm 파일을 읽어서 model에 들어갈 수 있게 npy파일로 바꾸는 역할  
 data_read_mat.py  -> mat 파일을 읽어서 model에 label로 들어갈 수 있게 npy파일로 바꾸는 역할  
 dataset.py -> 데이터 로더 구현과 transform 구현한 파일  
 image.py -> 파일이 잘 만들어졌는지 확인하는 파일  
 model.py -> unet이 구현되어 있음.  
 resize.ipynb -> resize를 위해 연습한 것( 필요없음)  
 results_data_name.py -> 만들어진 data는 output_000 꼴로 되어 있는데 이를 처음 준 데이터 이름과 일치하게 바꾸어주는 파일  
 run.ipynb -> colab에서 돌리기 위해 만든 파일  
 train.py -> train시와 test 시의 과정이 저장되어 있음  
 util.py -> cheakpoint를 저장하고 불러오는 함수가 저장되어 있음( model weight)  
 train -> data_read_dcm.py  data_read_mat.py dataset.py image.py model.py run.ipynb train.py  util.py  
 test ->  data_read_dcm.py dataset.py model.py results_data_name.py run.ipynb   train.py  util.py   
 
- model weight : 학습한 모델을 불러오기 위한 파일로 cheakpoint_512에 저장되어 있음 final에 사용한 데이터 모습은 cheakpoint_512 파일 중에서도 epoch 1000에 해당한 모델임 . 그냥 cheakpoint 파일명에 들어있는 건 resize를 256x256한걸로 final에서는 사용하지 않음 (cheakpoint_512 는 512x512 로 resize후 unet에 통과한 모델이 들어있다.)  
 util.py 파일에서 load를 수정하면 다른 cheakpoint로 이동 가능 (ex -1 -> -2 이런식으로 index를 수정) 단, 현재는 다 지우고 1000 epoch만 남겨둠

- 결과 파일 일단은 results/numpy 폴더에 저장되어 있음


만약 이 파일의 보고서를 확인하고 싶으시면 아래 링크에서 확인해주세요
