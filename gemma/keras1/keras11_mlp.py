# MLP(Multi Layer Perceptron)
# 열이 두 개 이상, 즉 다차원의 경우를 알아보자

# 1. 데이터 구성
import numpy as np
from sklearn.model_selection import train_test_split

# xxx = np.array([range(10), range(11, 21)])
# yyy = np.array([range(10), range(11, 21)]) # 2행 10열... 10행 2열로 바꿔야함 -> 전치

# xxx = np.array([range(100), range(311, 411)])
# yyy = np.array([range(501, 601), range(611, 711)])

# 열이 세 개인 경우
xxx = np.array([range(100), range(311, 411), range(201, 301)])
yyy = np.array([range(501, 601), range(611, 711), range(101,201)])


xxx = np.transpose(xxx) # 전치행렬
print(xxx)
print(xxx.shape)

yyy = np.transpose(yyy) 
print(yyy)
print(yyy.shape)  # 10행 2열이 됨, 즉 2차원

'''
model.add(Dense(200, input_dim=1, activation='relu')) 을
model.add(Dense(200, input_shape=(1,), activation='relu')) 로 변경해도 문제 없음
'''

x_train, x_test, y_train, y_test = train_test_split(xxx, yyy, test_size=0.2, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=66)

print("train shape: ", x_train.shape)

# 2. 모델 구성
from keras.models import Sequential 
from keras.layers import Dense
model = Sequential()

model.add(Dense(200, input_dim=3, activation='relu'))  
model.add(Dense(30))
model.add(Dense(100))
model.add(Dense(333))
model.add(Dense(133))
model.add(Dense(22))
model.add(Dense(777))
model.add(Dense(20))
model.add(Dense(3))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=300, batch_size=1, validation_data=(x_val, y_val))

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
xxx = np.array([range(100), range(311, 411)])
yyy = np.array([range(501, 601), range(611, 711)]) 로 데이터를 바꿔보자
    RMSE를 최대한 낮추려면? 0.00 이하로
     -> 노드의 수를 줄이고 epoch 값을 늘린다


xxx = np.array([range(100), range(311, 411)])
yyy = np.array([range(501, 601), range(111, 11, -1)]) 로 데이터를 바꿔보자


열이 세 개인 경우
xxx = np.array([range(100), range(311, 411), range(201, 301)])
yyy = np.array([range(501, 601), range(611, 711), range(101,201)])

model.add(Dense(200, input_dim=3, activation='relu'))  # input_dim=3
model.add(Dense(30))
model.add(Dense(100))
model.add(Dense(333))
model.add(Dense(133))
model.add(Dense(22))
model.add(Dense(20))
model.add(Dense(3))
'''