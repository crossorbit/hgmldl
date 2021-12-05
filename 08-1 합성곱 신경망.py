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

# +
# Convolutional Neural Network
from tensorflow import keras
from sklearn.model_selection import train_test_split

# 일반 이미지는 깊이(채널) 차원이 있음. 흑백 이미지는 채널 차원이 없는 2차원. 그래서 Conv2D 층 사용을 위해서 채널 차원 추가.
(train_input, train_target), (test_input, test_target) =\
    keras.datasets.fashion_mnist.load_data()
train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)
# -

# Sequential 클래스의 객체를 만들고, 첫 번째 합성곱 층이 Conv2D 추가(add 메서드 활용)
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28, 28, 1)))

# 풀링 층 추가. 전형적인 풀링 크기인 (2,2) 사용
# 패선 MNIST 이미지 (28, 28) 크기에 세임 패딩을 적용하면 특성 맵의 크기는 절반으로 줄어듬.
# 합성곱 층에 32개 필터 사용했기 때문에 특성맵의 크기는 32. 따라서 최대 풀링 통과한 특성맵은 (14,14,32)가 됨
model.add(keras.layers.MaxPooling2D(2))
model.summary()

# 첫번째 합성곱-풀링 층 다음에 두번째 합성곱-풀링 층 추가하자. 필터 개수는 64개로 늘려보자
model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2))
model.summary()

# 이제 3차원 특성맵을 일렬로 펼칠 차례! 왜? 마지막에 10개의 뉴런을 가진 출력층에서 확률 계산하기 때문.
# Flattern 으로 펼치고, Dense 은닉층과 Dropout 추가, 마지막으로 Dense 출력층 구성
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

# +
# plot_model : 합성공 신경망 모델의 층 구성을 그림으로 표현 
import pydot as pyd
from keras.utils.vis_utils import plot_model

keras.utils.plot_model(model, show_shapes=True)
# -

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
history = model.fit(train_scaled, train_target, epochs=20,
                   validation_data=(val_scaled, val_target),
                   callbacks=[checkpoint_cb, early_stopping_cb])

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

model.evaluate(val_scaled, val_target)

plt.imshow(val_scaled[0].reshape(28, 28), cmap='gray_r')
plt.show()

preds = model.predict(val_scaled[11:12])
print(preds)

plt.bar(range(1, 11), preds[0])
plt.xlabel('class')
plt.ylabel('prob.')
plt.show()

classes = ['티셔츠', '바지', '스웨터', '드레스', '코트',
           '샌달', '셔츠', '스니커즈', '가방', '앵클 부츠']

import numpy as np
print(classes[np.argmax(preds)])

test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0
model.evaluate(test_scaled, test_target)


