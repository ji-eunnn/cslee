# validation(검증)

# 1. 데이터 구성
import numpy as np
x_train = np.array([1,3,4,5,6,7,9,10]) 
y_train = np.array([1,3,41,5,6,7,9,10])

x_test= np.array([1001,1002,1004,1006,1007,1008,1009,1010])
y_test= np.array([1001,1002,1004,1006,1007,100,1009,1010])

x_val = np.array([101,102,103,104,105])
y_val = np.array([101,102,103,104,105])  # 검증용 데이터셋 생성


# 2. 모델 구성
from keras.models import Sequential 
from keras.layers import Dense
model = Sequential()

model.add(Dense(200, input_dim=1, activation='relu'))
model.add(Dense(305))
model.add(Dense(1230))
model.add(Dense(20))
model.add(Dense(4))
model.add(Dense(444))
model.add(Dense(111))
model.add(Dense(660))
model.add(Dense(10))
model.add(Dense(789))
model.add(Dense(22))
model.add(Dense(1))

# 3. 훈련
model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=400, batch_size=1, validation_data=(x_val, y_val))
'''
validation(검증) 추가 : validation_data=(x_test, y_test) => 결과는 조금 더 좋아졌지만 문제점은? 과적합 => 다른 데이터를 넣어주어야 함
=> x_val, y_val 생성해서 validation_data 값 수정(x_test, y_test -> x_val, y_val)
=> 결과가 더 좋아짐 왜? 훈련도가 높아졌기 때문

'''

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

'''
데이터를 수정해보자
    x_train = np.array([1,22,3,4,5,6,7,8,9,10]) 
    y_train = np.array([1,2,3,41,5,6,7,28,9,10])

    x_test= np.array([1001,1002,1003,1004,15,1006,1007,1008,1009,1010])
    y_test= np.array([1001,1002,103,1004,1005,1006,1007,100,1009,1010])
-> R2 값 확 낮아짐
-> 노드의 갯수, 깊이, epochs, batch_size 등을 바꿔서 R2 값을 높여보자 -> 한계가 있음

이상 데이터를 20%만 지워보자
    x_train = np.array([1,3,4,5,6,7,9,10]) 
    y_train = np.array([1,3,41,5,6,7,9,10])

    x_test= np.array([1001,1002,1004,1006,1007,1008,1009,1010])
    y_test= np.array([1001,1002,1004,1006,1007,100,1009,1010])
=> 아주 기초적인 전처리로 결과값이 좀 더 좋아짐

'''

'''
x_train : 훈련(by 머신)
x_val : 검증(by 머신)
x_test : 검증(by 사람)
'''

'''
+ 주가 예측
x : 어제 주가, 날씨 등등 / y : 주가          알고자 하는 것 : 내일의 주가(y)
10년치 데이터가 있다 -> 5년치 데이터로 train, 2년치 데이터로 validation -> 3년치 데이터를 집어넣어서 평가, 예측 -> y_test와 y_predict 서로 비교: R2, RMSE
-> 결과가 좋다? 그 모델에 완전히 새로운 데이터 즉, 오늘의 주가, 날씨 등등을 넣어서 내일의 주가를 예측
'''