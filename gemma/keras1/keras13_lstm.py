# 순환신경망 : Recurrent Neural Network (RNN) - 시계열

from numpy import array
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 1. 데이터 구성
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = array([4,5,6,7])

print("x.shape : ", x.shape)    # (4, 3)
print("y.shape : ", y.shape)    # (4,)

# reshape from [samples, timesteps] into [samples, timesteps, features]
x = x.reshape((x.shape[0], x.shape[1], 1))  # (x.shape[0], x.shape[1], 1) = (4, 3, 1)   1: 몇 개씩 작업하는가 like batch size

print("x.shape : ", x.shape)    # (4, 3, 1)
print("y.shape : ", y.shape)    # (4,)


# 2. 모델 구성
model = Sequential()
model.add(LSTM(200, activation= 'relu', input_shape=(3,1)))  # input_shape=(4,3,1) -> (3,1) 행무시
model.add(Dense(10))
model.add(Dense(85))
model.add(Dense(792))
model.add(Dense(2212))
model.add(Dense(2))
model.add(Dense(1000))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse')
# 첫 단은 LSTM을 적용, 다음 단부터는 dense로 적용했는데도 좋은 결과

# 3. 실행
model.fit(x, y, epochs = 1000, verbose=2)
# demonstate prediction
# x_input = array([6,7,8])
x_input = array([70,80,90])
x_input = x_input.reshape((1,3,1))
yhat = model.predict(x_input, verbose=0)
print(yhat)

'''
70, 80, 90을 넣어보자
x_input = array([70,80,90]) -> 결과 : 83.77
    왜? 정제된 데이터긴 하지만 새로운 데이터에 도움이 안됨
'''

'''
x : (5, 3)  y : (1,)
  x   y
1 2 3 4
2 3 4 5
3 4 5 6
4 5 6 7
5 6 7 8
    DNN의 경우 -> input_dim = 3
    RNN의 경우 -> reshape해야함 x = (5, 3, 1) 이 때 행무시, input_shape=(3, 1)
'''