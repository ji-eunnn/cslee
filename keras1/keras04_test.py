# x1과 y1으로 훈련

# 1. 데이터 구성
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) 
y_train = np.array([1,2,3,4,5,6,7,8,9,10]) # 데이터의 양을 늘림

x_test= np.array([11,12,13,14,15,16,17,18,19,20])
y_test= np.array([11,12,13,14,15,16,17,18,19,20])
# train과 test로 분리한 이유? 더 좋은 결과를 위해.. 

# 2. 모델 구성
from keras.models import Sequential 
from keras.layers import Dense
model = Sequential()

model.add(Dense(2000000, input_dim=1, activation='relu')) 
model.add(Dense(4))
model.add(Dense(12))
model.add(Dense(6))
model.add(Dense(1))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=30, batch_size=2) # 한 batch의 사이즈가 2고, 데이터 수가 10개니까 한 에포당 5번 작업, 총 5*30=150개/즉 batch는 일괄 작업
model.fit(x_train, y_train, epochs=30) # acc:0.0 ? batch size default값이 32니까 데이터 이상의 수 -> 제대로 작동하지 않음

# 4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("acc : ", acc)