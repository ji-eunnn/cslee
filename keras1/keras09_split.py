# 훈련용 데이터와 검증용 데이터를 분리하는 함수

# 1. 데이터 구성
import numpy as np

xxx = np.array(range(100))  # 0~99
yyy = np.array(range(100))

'''
x_train 60%
y_train 60%
x_val 20%
y_val 20%
x_test 20% 
y_test 20%
    ~로 분리하는 함수를 작성해보자
'''
x_train = xxx[:60]
y_train = yyy[:60]
x_val = xxx[60:80]
y_val = yyy[60:80]
x_test = xxx[80:]
y_test = yyy[80:]  # 리스트 형태로 분리

print("x_train.shape", x_train.shape)
print("x_val.shape", x_val.shape)
print("x_test.shape", x_test.shape)

#2. 모델 구성
from keras.models import Sequential 
from keras.layers import Dense
model = Sequential()

model.add(Dense(200, input_dim=1, activation='relu'))
model.add(Dense(30))
model.add(Dense(100))
model.add(Dense(20))
model.add(Dense(1))

# 3. 훈련
model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

# 4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("acc : ", acc)
print("loss : ", loss)

y_predict = model.predict(x_test)
print(y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score 
r2_y_predict = r2_score(y_test, y_predict)  
print("R2 : ", r2_y_predict)
