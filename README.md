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


만약 이 파일의 보고서를 확인하고 싶으시면 이관희_2017142136_최종결과.pdf 파일을 참고해주세요


# 만약 다른 프로젝트에서 사용하기 위헤서 참고할 떄 팁
먼저 코드를 이해하시고 그 후에 반드시 정확도를 올리기 위한 작업을 해야합니다. 왜냐하면 이 모델 자체는 단순히 주어진 데이터 형태를 가공해서 unet에 넣기 좋게 변형하고 결과를 뽑아내는 작업이지
절대 정확도 높은 결과가 아닙니다. 물론 이렇게 돌리더라도 정확도 판단 기준에 따라서 평균 70%의 정확도를 도출하는 것을 확인했지만 이것으로는 부족할 것입니다.
시간이 있으시다면 먼저 input 이미지를 정형화해주는 전처리가 필요합니다. 여기서 전처리가 굉장히 어렵게 느껴질 수도 있는데 사실 그 개념은 간단히 model을 통과하는 데이터를 일정하게 만들어주는 것에 있습니다.
- 예를 들면 9가 회전되어 있는 형태를 classification 할 때 우리는 모든 9의 형태를 집어넣어주는 것보다 사실 model input에 들어갈 때 변형된 9을 학습된 데이터와 맞게 회전을 시키는 등의 방법으로 맞추어 주는 것이 중요합니다.
- lumbar의 경우 혹은 , 다른 medical image의 경우 밝기나 찍은 방향 (L,R) 그리고 혹은 데이터 첼린지를 위해 회전한 형태도 있을 수 있습니다. 만약 데이터가 어느정도 있다면 학습한 데이터도 모두 같은 조건으로 맞추어주고 학습을 시키고 결과 데이터 역시 model의 input으로 들어갈때 같은 조건으로 맞출 수만 있다면 정확도는 매우 올라가게 됩니다. 
- 이런 조건을 맞추는 방법은 디지털 신호 처리를 배우면 알 수도 있지만 기본적으로는 공부가 필요합니다. (다양한 transform을 말하는 거임)
- 만약 다시 저한테 이 task가 주어진다면 밝기 같은 것에 집중하기 보다는 적어도 회전과 반전을 통일시켜줄 것 같긴합니다
- edge detector을 이용한 방법은 x- lay의 경우 원하지 않은 다른 부분 역시 noise로 검출되기 때문에 추천하지 않습니다. 

설치해야 하는 파일은 잘 설치하시고 빨간줄 뜨는 거 모두 깔면 됩니다. 
이 프로젝트는 2021 - 11 월에 진행한 것으로 torch는 그 당시 최신버전으로 설치했었습니다. 