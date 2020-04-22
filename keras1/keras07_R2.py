# R2 : 결정계수

# 1. 데이터 구성
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) 
y_train = np.array([1,2,3,4,5,6,7,8,9,10])

x_test= np.array([1001,1002,1003,1004,1005,1006,1007,1008,1009,1010])
y_test= np.array([1001,1002,1003,1004,1005,1006,1007,1008,1009,1010])

'''
x3 = np.array([101,102,103,104,105])
x4 = np.array(range(30, 50))
'''

# 2. 모델 구성
from keras.models import Sequential 
from keras.layers import Dense
model = Sequential()

model.add(Dense(200, input_dim=1, activation='relu')) # 활성화 함수 : 'relu' -> 'linear' 좋은 결과 but linear보단 relu가 조금 더 성능 좋음
model.add(Dense(305))
model.add(Dense(1230))
model.add(Dense(60))
model.add(Dense(22))
model.add(Dense(1))
# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) # 최적화 함수 : 'adam' -> 'rmsprop', 'sgd'
# loss = 'mse', 'mean_square_error'-> 'mae', 'mean_absolute_error' 변경 가능하지만 지표 엉망
# metrics = 'accuracy'-> 'mae' 변경 가능하지만 지표 엉망
model.fit(x_train, y_train, epochs=200, batch_size=2) 

# 4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("acc : ", acc)
print("loss : ", loss)  # 0.25

y_predict = model.predict(x_test)
print(y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# 새로운 평가지표 : R2 구하기
from sklearn.metrics import r2_score 
r2_y_predict = r2_score(y_test, y_predict)  # 훈련을 시켜서 나온 값과 원값을 서로 비교
print("R2 : ", r2_y_predict)

