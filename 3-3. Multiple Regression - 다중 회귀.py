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

# !pip install pandas

import pandas as pd
df = pd.read_csv('http://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
print(perch_full)

import numpy as np
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
1000.0])

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)

from sklearn.preprocessing import PolynomialFeatures

# fit: 새롭게 만들 특성 조합을 찾음, transform: 실제로 데이터를 변환함
# 왜 2,3 이라는 2개의 특성이 1,2,3,4,6,9 라는 6개의 특성으로 변환되었을까?
# PolynomialFeatures 은 입력된 값의 제곱항(2*2 = 4, 3*3 = 9) 과 두 입력의 곱(2*3 =6), 그리고 1을 추가함
# 그러나 사이킷런의 선형 모델은 자동으로 절편을 추가하므로 굳이 이렇게 만들 필요 없음. include_bias=False 로 지정하자 
poly = PolynomialFeatures()
poly.fit([[2,3]])
print(poly.transform([[2,3]]))

# 그러나 사이킷런의 선형 모델은 자동으로 절편을 추가하므로 굳이 이렇게 만들 필요 없음. include_bias=False 로 지정하자 
# 꼭 지정하지 않아도 사이킷런 모델은 자동으로 추가된 절편항(1)을 무시해서 명시하지 않아도 됨
poly = PolynomialFeatures(include_bias=False)
poly.fit([[2,3]])
print(poly.transform([[2,3]]))

poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_poly.shape)

# get_feature_names: 어떤 9개의 특성이 만들어졌을까?
poly.get_feature_names()

test_poly = poly.transform(test_input)

# 다중 회귀 모델 훈련하기
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)

print(lr.score(train_poly, train_target))

print(lr.score(test_poly, test_target))

# degree 매개변수를 활용하여 필요한 고차항의 최대차수 지정 가능
poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape)

# 이 데이터를 가지고 다시 훈련 해보자
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))

# 특성 갯수가 많아지면 훈련 세트에 너무 과적합 되어 테스트 세트에 나쁜 결과를 야기함
print(lr.score(test_poly, test_target))

# StandardScaler 사용하여 규제regularization 적용 해보자
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# 릿지ridge : 계수를 제곱한 값을 기준으로 규제 적용, 라쏘lasso : 계수의 절대값을 기준을 적용
# 보통 릿지를 선호함. 라쏘는 잘못하면 계수의 크기가 0으로 적용됨
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))

print(ridge.score(test_scaled, test_target))

# 하이퍼파라미터 : 머신러닝 모델이 학습할 수 없고 사람이 알려줘야 하는 파라미터
# 릿지, 라쏘 모델은 규제의 양을 임의로 조정할 수 있음 -> alpha 매개변수로 규제의 강조 조절
import matplotlib.pyplot as plt
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # 릿지 모델 만들기
    ridge = Ridge(alpha=alpha)
    # 릿지 모델 훈련
    ridge.fit(train_scaled, train_target)
    # 훈련세트와 테스트 세트 저장
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))

plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

 # 가장 가깝고 테스트 점수가 높은 -1(10의 -1승 = 0.1) 기준으로 최종 모델 훈련
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))


