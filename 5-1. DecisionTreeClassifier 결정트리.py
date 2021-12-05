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

import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')
wine.head()

wine.info()

wine.describe()

#pandas 데이터프레임을 numpy 배열로 바꾸자
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# +
#train_test_split 은 기본 25%로 테스트 세트 지정. 변경 시 test_size 활용
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
# -

print(train_input.shape, test_input.shape)

#StandardScaler 활용한 훈련 세트 전처리 
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

#로지스틱회귀 모델 훈련해보자
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

print(lr.coef_, lr.intercept_)

# 위 모델은 이유를 설명하기 어려움. 그래서 DecisionTreeClassifier 를 사용해보자
# 훈련세트 점수 >> 테스트세트 점수 : 과대적합#
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

# plot_tree() 함수 사용하여 그려보자
# figsize : 창의 크기 지정 가로 10인치, 세로 7인치
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()

# 위 그림은 너무 복잡하나ㅣ, 트리의 깊이를 제한해서 출력해보자
# max_depth = 1 이면, 루트 노드를 제외하고 하나의 노드를 더 확장해 그림
# filled 는 클래스에 맞게 노드의 색을 칠할 수 있음
# feature_names : 특성의 이름을 전달할 수 있음
plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

# 의사결정트리를 가지치기 해보자
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

# 전처리 하지 않은 데이터 train_input 기준으로 훈련해보자
# 결과가 동일하게 나온다!
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

# 결과는 같지만, 표준화하지 않은 특성 값이여서 이해하기 쉽다
plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

print(dt.feature_importances_)


