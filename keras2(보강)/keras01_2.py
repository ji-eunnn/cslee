
# 1. 데이터 구성
import numpy as np
x = np.array(range(1,11)) 
y = np.array(range(1,11))
x2 = np.array([4,5,6])

# 2. 모델 구성
from keras.models import Sequential  # 순차적 모델을 위해 케라스에 있는 모델을 임폴트
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim=1, activation='relu'))  # 노드의 개수.. input이 1개, output이 5개인 레이어
model.add(Dense(100))
model.add(Dense(555))
model.add(Dense(4))
model.add(Dense(1))


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])   # 최적의 weight를 위한 parameter값이 adam
model.fit(x, y, epochs=100, batch_size=1)  # 100번 훈련시키고, 하나씩 잘라서 작업하라, loss값 점차 떨어짐

# 4. 평가 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print("acc : ", acc)

y_predict = model.predict(x2)
print(y_predict)





