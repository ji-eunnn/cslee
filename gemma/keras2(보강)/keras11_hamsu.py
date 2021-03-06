# 1. 데이터 구성
import numpy as np
x_train = np.array(range(1,11)) 
y_train = np.array(range(1,11))
x_test = np.array(range(11,21))
y_test = np.array(range(11,21))
x3 = np.array(range(101, 107))
y3 = np.array(range(30, 50))



# 2. 모델 구성2 -> 앙상블하기 쉬움!
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()

input1 = Input(shape=(1, ))
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(22)(dense1)
dense3 = Dense(55)(dense2)
output1 = Dense(1)(dense3)

model = Model(inputs=input1, outputs = output1)
model.summary()


# 3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=100)

# # 4. 평가 예측
# loss, acc = model.evaluate(x_test, y_test, batch_size=1)  
# print("acc : ", acc)

# y_predict = model.predict(x_test)
# print(y_predict)


# # RMSE 구하기
# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print("RMSE : ", RMSE(y_test, y_predict))


# # R2 구하기
# from sklearn.metrics import r2_score
# r2_y_predict = r2_score(y_test, y_predict)
# print("R2 : ", r2_y_predict)

