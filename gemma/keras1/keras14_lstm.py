import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
# from keras.utils import np_utils
# a = np.array([11,12,13,14,15,16,17,18,19,20])
a = np.array(range(11,21))
# a = np.array(range(100))
window_size = 5
def split_5(seq, window_size):  # 데이터를 5개씩 자르기용.    # 입력이 5이고 5개씩 자르기
    aaa = []
    for i in range(len(a)-window_size +1):                 # 열
        subset = a[i:(i+window_size)]       # 0~5
        aaa.append([item for item in subset])
        # print(aaa)
    print(type(aaa))    
    return np.array(aaa)

dataset = split_5(a, window_size)     # 5씩 잘랏으니 (5, 6)가 된다. // window_size+1 만큼씩 잘라진다.
print("===========================")
print(dataset)
'''
[[11 12 13 14 15]
 [12 13 14 15 16]
 [13 14 15 16 17]
 [14 15 16 17 18]
 [15 16 17 18 19]
 [16 17 18 19 20]]
'''
print(dataset.shape)    # (6, 5)


#입력과 출력을 분리시키기  5개와 1개로

x_train = dataset[:,0:4]
y_train = dataset[:,4]

print(x_train)
'''
[[11 12 13 14]
 [12 13 14 15]
 [13 14 15 16]
 [14 15 16 17]
 [15 16 17 18]
 [16 17 18 19]]
 '''

x_train = np.reshape(x_train, (len(a)-window_size+1, 4, 1))
print(x_train)
'''
[[[11]
  [12]
  [13]
  [14]]

 [[12]
  [13]
  [14]
  [15]]

 [[13]
  [14]
  [15]
  [16]]

 [[14]
  [15]
  [16]
  [17]]

 [[15]
  [16]
  [17]
  [18]]

 [[16]
  [17]
  [18]
  [19]]]
'''

x_test = np.array([[[21],[22],[23],[24]], [[22],[23],[24],[25]], 
                  [[23],[24],[25],[26]], [[24],[25],[26],[27]]])
y_test = np.array([25, 26, 27, 28])
print(x_test)
'''
[[[21]
  [22]
  [23]
  [24]]

 [[22]
  [23]
  [24]
  [25]]

 [[23]
  [24]
  [25]
  [26]]

 [[24]
  [25]
  [26]
  [27]]]
'''
print(y_test) # [25 26 27 28]

print(x_train.shape)    # (6, 4, 1)
print(y_train.shape)    # (6, )
# print(x_test.shape)     # (4, 4, 1)
# print(y_test.shape)     # (4, )

print(x_train)

# 모델 구성하기
model = Sequential()
model.add(LSTM(32, input_shape=(4,1), return_sequences=True))
model.add(LSTM(10))

# model.add(Dropout(0.2))
model.add(Dense(5, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(78, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1111, activation='relu'))
model.add(Dense(287, activation='relu'))
model.add(Dense(34, activation='relu'))
model.add(Dense(791, activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=300, batch_size=1, verbose=2)

loss, acc = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)  
y_predict2 = model.predict(x_train)

print('loss : ', loss)
print('acc : ', acc)
print('y_predict(x_test) : \n', y_predict)
    # 25, 26, 27, 28이 나와야함
    # 20.022, 20.023, 20.025, 20.026 왜 이런 결과가? 데이터가 10개뿐.. split해서 늘린 아주 작은 데이터
    # LSTM층이 데이터에 비해 너무 깊음 -> 과적합 -> 차라리 LSTM층을 줄이자
# print('y_predict2(x_train) : \n', y_predict2)
    # 15.03, 15.96, 17.00, 17.98, 18.93, 19.97 (행 개수는 무시) 괜찮은 결과 왜? 훈련한 데이터이기 때문



'''
      x     y
21 22 23 24 25
22 23 24 25 26
23 24 25 26 27
...

'''