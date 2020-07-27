## IITP 이미지 내 쓰러진 사람 영역 검출  

Task
```
이미지 내의 쓰러진 사람 영역 검출 
```

Dataset Description:
```
\_train
    \_ train_data (folder)
        \_ dummy_file (no training data)
\_test
    \_ test_data (folder)
        \_ *.jpg 
    \_ test_label 
\_test_submit
    \_ test_data (folder)
        \_ *.jpg 
    \_ test_label 

# of test images: 34,822 
# of bounding boxes: 8,039
```

Input data description:
```
- 5 fps sampling from the original videos (350 clips with 20 seconds).
- image resolution: [960 x 540] (down-sampled from original [1920 x 1080] resolution) 
```

Output format:
```
test dataset 중 쓰러진 사람이 검출된 이미지 (ex: 'a.jpg', 'b.jpg', ...) 에 대해서만 
결과를 저장하여 다음과 같이 tuple 형태로 infer function 에서 return 한다.  
    (['a.jpg', 'b.jpg', ...], 
    [[a_x1, a_y1, a_x2, a_y2], [b_x1, b_y1, b_x2, b_y2], ...]) 
쓰러진 사람이 검출되지 않은 이미지는 출력 값에 포함시키지 않는다.   

Bounding box [x1, y1, x2, y2] 는 실제 이미지 픽셀 크기에 해당하는 값이며 int type 을 사용하여 내보낸다.
이미지마다 최대 한 개의 bounding box 를 내보낼 수 있다.
```


Evaluation Metric (Recall):
```
test dataset 중 쓰러진 사람이 존재하는 이미지에 대해서만 evaluation 수행. 
Ground truth 와 predicted bounding box 의 IoU (intersection-over-union) 이 0.5 이상인 경우 검출 성공으로 판단.
Recall = (검출 성공한 이미지 수) / (쓰러진 사람이 존재하는 이미지 수)
```


Local model 로드:

본 baseline에서는 main.py와 같은 위치에 'model.pth' 이라는 pre-trained 모델이 있다면 해당 파일을 [코드상에서 load 한다.](https://github.com/AI-GrandChallenge/round-1/blob/master/iitp_falling/main.py#L154)

마찬가지로 본 baseline을 따라 `nsml run`을 이용해 main.py를 실행할때 pre-trained model 을 올리고(main.py와 같은 폴더에 있는 파일을 전부 올림), 

코드상에서 weight load 를 한후에 nsml.save()를 이용해서 model 을 [세션안에 저장한다.](https://github.com/AI-GrandChallenge/round-1/blob/master/iitp_falling/main.py#L190)

submit을 할때는 [여기](https://github.com/AI-GrandChallenge/round-1/blob/master/iitp_falling/main.py#L193) 에서 nsml.bind 에 bind 된 [load함수가 호출된후](https://github.com/AI-GrandChallenge/round-1/blob/master/iitp_falling/main.py#L151)

[bind된 infer 함수가 실행되게 된다](https://github.com/AI-GrandChallenge/round-1/blob/master/iitp_falling/main.py#L148) (nsml 에서 자동으로 실행함)


How to run:

```bash
nsml run -d tr-1 -e main.py -c 2 --memory 16G
```

How to list checkpoints saved:

```bash
nsml model ls {USER_ID}/{CHALLENGE_NAME}/{SESSION_NUMBER}
```

How to submit:

```bash
nsml submit  {USER_ID}/{CHALLENGE_NAME}/{SESSION_NUMBER} {CHEKCOPOINT_NAME}
```
