# RMSE : 평균 제곱근 오차

# 1. 데이터 구성
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) 
y_train = np.array([1,2,3,4,5,6,7,8,9,10])

x_test= np.array([11,12,13,14,15,16,17,18,19,20])
y_test= np.array([11,12,13,14,15,16,17,18,19,20])

x3 = np.array([101,102,103,104,105])
x4 = np.array(range(30, 50))

# 2. 모델 구성
from keras.models import Sequential 
from keras.layers import Dense
model = Sequential()

model.add(Dense(200, input_dim=1, activation='relu')) 
model.add(Dense(35))
model.add(Dense(125))
model.add(Dense(64))
model.add(Dense(1))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30, batch_size=2) 

# 4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("acc : ", acc)
print("loss : ", loss)  # 0.25

y_predict = model.predict(x_test)
print(y_predict)


# RMSE라는 평가지표를 새로 추가해보자 acc, predict, mse, rmse
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) :   # 훈련을 시켜서 나온 값과 원값을 서로 비교
    return np.sqrt(mean_squared_error(y_test, y_predict))  
print("RMSE : ", RMSE(y_test, y_predict))  # RMSE 값을 최대한 작게 