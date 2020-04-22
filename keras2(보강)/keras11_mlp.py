# 1. 데이터 구성
import numpy as np

x = np.array([range(100), range(311, 411), range(100)])
y = np.array([range(500, 600), range(711, 811), range(100)])    # w가 5..., 2...., 1 로 섞여있으므로 rmse 개똥으로 나온다

x = np.transpose(x)
y = np.transpose(y)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=66) 
x_val, x_test, y_val, y_test = train_test_split(x, y, test_size=0.5, random_state=66)



# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim=3, activation='relu'))
model.add(Dense(3))
model.add(Dense(100))
model.add(Dense(4))
model.add(Dense(3))


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100)

# 4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)  
print("acc : ", acc)

y_predict = model.predict(x_test)
print(y_predict)


# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))