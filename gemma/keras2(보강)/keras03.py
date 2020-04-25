
# 1. 데이터 구성
import numpy as np
x = np.array(range(1,11)) 
y = np.array(range(1,11))
x2 = np.array([4,5,6])

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

model.summary()


# # 3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.fit(x, y, epochs=100)

# # 4. 평가 예측
# loss, acc = model.evaluate(x, y, batch_size=3)  # 배치 사이즈를 3으로 바꿔봄 -> acc 떨어짐 ? (1,2,3), (4,5,6)... 데이터에 비해 너무 큰 배치사이즈
# print("acc : ", acc)

# y_predict = model.predict(x2)
# print(y_predict)





