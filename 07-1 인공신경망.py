# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

from tensorflow import keras
(train_input, train_target), (test_input, test_target) =\
    keras.datasets.fashion_mnist.load_data()

print(train_input.shape, train_target.shape)

print(test_input.shape, test_target.shape)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 10, figsize=(10, 10))
for i in range(10):
    # 일반적으로 픽셀(0~255) 기준으로 숫자가 클수록 검으색에 가까워짐 
    # 0에 가까울수록 검게 보이도록 gray reverse scale 적용
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')
plt.show()

print([train_target[i] for i in range(10)])

# 넘파이의 unique 함수를 사용하여 레이블당 샘플 갯수 확인. 6000개로 균일한것 볼수 있음
import numpy as np
print(np.unique(train_target, return_counts=True))

# ㅁ 로지스틱 회귀 모델 사용하여 훈련해보자
#  - 훈련 샘플이 6만개나 되기 때문에 한번에 훈련하기 보다는, 샘플을 하나씩 꺼내서 훈련하는 방법이 더 효율적
#  - 바로 확률적 경사하강법 SGDClassifier. loss 매개변수를 'log'로 지정
#  - 이것을 사용할 때는 표준화된 전처리 데이터를 사용함.

# +
# 패션 MNIST의 픽셀은 0~255 사이의 정수값을 가짐. 그래서 255로 나눠서 0~1사이로 정규화 함
# -> 표준화는 아니지만 양수값으로 이미지를 전처리 할 때 많이 사용하는 방법
# SGDClassifier는 2차원 입력 다루지 못하기 때문에 reshape 메소드를 활용하여 2차원 샘플 을 1차원으로 펼치자.

train_scaled = train_input / 255.0
print('원본', train_scaled.shape)

train_scaled = train_scaled.reshape(-1, 28*28)

print('1차원', train_scaled.shape)
# -

from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
# 2개의 분류에서 log함수 쓰면 하나는 양성 나머지는 음성으로 훈렴함 -> 1개의 방정식을 시그모이드 로 바꿈
# 10개의 다중 분류에서 log함수 쓰면 (하나는 양성 나머지 9개는 음성) * 10개 Case 훈련함 -> 10개의 선형방정식을 softmax로 확률로 바꿈
sc = SGDClassifier(loss='log', max_iter=9, random_state=42)
scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
print(np.mean(scores['test_score']))

# ㅁ 텐서플로수 = 케라스
#  - 대표적 딥러닝 라이브러리. 그외는 페이스북이 만든 Pytorch가 있음
#  - 딥러닝 라이브러리는 다른 머신러닝 라이브러리와 달리 그래픽 처리 장치인 GPU 사용하여 인공신경망 훈련함
#    GPU는 벡터와 행렬 연산에 최적화되어 있어서 곱셈과 덧셈을 많이하는 인공신경망에 큰 도움이 됨

import tensorflow as tf
from tensorflow import keras

# +
# 딥러닝에서는 교차검증 사용하지 않고 검증세트를 별도로 덜어내어 사용함.
# 왜? 딥러닝 분야의 데이터셋은 충분히 커서 검증 점수가 안정적. 그리고 교차검증 하기에는 훈련 시간이 너무 오래 걸림

from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)
# -

print(train_scaled.shape, train_target.shape)

print(val_scaled.shape, val_target.shape)

# 밀집층 dense Layer : 784개 픽셀과 10개 뉴런이 연결된 빽뺵한 선 (완전연결층)
# keras.layers.Dense(뉴런 개수, 뉴런의 출력에 적용할 함수, 입력의 크기)
# 10 : 10개의 패션아이템 분류
# activation='softmax' : 10개의 뉴런에서 출력되는 값을 확률로 바꾸기. 2개의 클래스 분류라면 activation='sigmoid'
# input_shape=(784,) : 10개의 뉴런이 각각 몇개의 입력을 받는지 튜플로 지정
dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))

# 앞서 정의한 밀집층을 가진 신경망 모델 만들기
model =  keras.Sequential(dense)

#케라스 모델 훈련 전 설정 단계가 있음. 
# 이진분류 : loss ='binary_crossentropy'
# 다중분류 : loss ='categorical_crossentropy'
# sparse_ 가 붙은 이유 : 데이터가 원-핫 인코딩 된 상태라면 불필요. 데이터 그대로 사용하려면 붙여서 원-핫 인코딩 효과 추가
# metrics='accuracy' : 정확도 함께 출력
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')

print(train_target[:10])

model.fit(train_scaled, train_target, epochs=5)

model.evaluate(val_scaled, val_target)


