# m03_xor.py를 케라스로 리폼

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from numpy import array
import keras


# 1. 데이터
x_data = array([[0,0], [1,0], [0,1], [1,1]])
y_data = array([0, 1, 1, 0])

print(x_data.shape)
print("========")

x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], 1))
print(x_data.shape)
print(y_data.shape)


# 2. 모델
from keras.models import Sequential
from keras.layers import LSTM, Dense
model = Sequential()

model.add(LSTM(5, input_shape=(2, 1), activation='relu')) 
model.add(Dense(30))
model.add(Dense(7))
model.add(Dense(1))

# 3. 실행
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])  
model.fit(x_data, y_data, epochs=100, batch_size=1)  


# 4. 평가 예측

x_test = array([[0,0], [1,0], [0,1], [1,1]])
x_test = x_data.reshape((x_data.shape[0], x_data.shape[1], 1))
y_predict = model.predict(x_test)

print(x_test, "의 예측결과 : ", y_predict)
print("acc ", accuracy_score([0,1,1,0], y_predict.round()))