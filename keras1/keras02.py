# Data -> Model -> Compile - Fit(train) -> Evaluate/Predict : 기계를 훈련시키는 4가지 단계

# 1. 데이터 구성
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10]) 
y = np.array([1,2,3,4,5,6,7,8,9,10]) # 데이터의 양을 늘림

# 2. 모델 구성
from keras.models import Sequential 
from keras.layers import Dense
model = Sequential()

model.add(Dense(2, input_dim=1, activation='relu')) 
model.add(Dense(143))
model.add(Dense(1526))
model.add(Dense(716))
model.add(Dense(188))
model.add(Dense(1))


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=200, batch_size=1) 

# 4. 평가 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print("acc : ", acc)