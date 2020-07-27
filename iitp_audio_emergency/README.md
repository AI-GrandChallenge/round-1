# IITP Track 2 - 음성인지 트랙
## Task 
- 음성분석을 통한 폭력 또는 위협 상황 검출하기 

## Dataset Description
```
\_ train
    \_ train_data (folder)
        \_ *.wav (dummy file)
    \_ train_label (dummy file)
\_ test
    \_ test_data (folder)
        \_ *.wav
    \_ test_label
\_ test_submit
    \_ test_data (folder)
        \_ *.wav
    \_ test_label 

# of test audio: 300
```

## Evaluation Metric
- 각 audio 별 IoU의 평균값 
- IoU = (intersection length) / (union length)
- 단, union length == 0 일 시, IoU = 1. (해당 음성 파일에 위협구간이 없는 경우)


## How to run 
```
nsml run -d tr-2 -e main.py -a "--num_epoch 0" 
```
You can modify the arguments when you want to change such as the number of epochs, learning rate, etc.
## How to list checkpoints saved
```
nsml model ls {USER_ID}/{CHALLENGE_NAME}/{SESSION_NUMBER}
```

## How to submit 
```
nsml submit {USER_ID}/{CHALLENGE_NAME}/{SESSION_NUMBER} {CHECKPOINT_NAME}
```
