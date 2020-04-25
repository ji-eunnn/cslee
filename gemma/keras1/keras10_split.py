# train_test_split 함수를 이용해 데이터를 분리해보자

# 1. 데이터 구성
import numpy as np
from sklearn.model_selection import train_test_split

xxx = np.array(range(100))  # 0~99
yyy = np.array(range(100))

x_train, x_test, y_train, y_test = train_test_split(xxx, yyy, test_size=0.2, random_state=66) # random_state 값을 주면 랜덤값이 변하지 않는다
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=66)
# xxx는 x_train과 x_test로 분리되고, yyy는 y_train, y_test로 분리된다

print("x_train.shape", x_train.shape)
print("x_val.shape", x_val.shape)
print("x_test.shape", x_test.shape)

print(x_train)
# print(y_train)
print(x_test)
# print(y_test)
print(x_val)
# print(y_val)

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
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
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

