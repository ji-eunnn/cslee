# 1. 데이터 구성
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) 
y_train = np.array([1,2,3,4,5,6,7,8,9,10]) # 데이터의 양을 늘림

x_test= np.array([11,12,13,14,15,16,17,18,19,20])
y_test= np.array([11,12,13,14,15,16,17,18,19,20])

x3 = np.array([101,102,103,104,105])
x4 = np.array(range(30, 50))

# 2. 모델 구성
from keras.models import Sequential 
from keras.layers import Dense
model = Sequential()

model.add(Dense(200, input_dim=1, activation='relu')) 
model.add(Dense(4))
model.add(Dense(12))
model.add(Dense(6))
model.add(Dense(1))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30, batch_size=2) # 한 batch의 사이즈가 2고, 데이터 수가 10개니까 한 에포당 5번 작업, 총 5*30=150개/즉 batch는 일괄 작업
# model.fit(x_train, y_train, epochs=100) # acc:0.0 ? batch size default값이 32니까 데이터 이상의 수 -> 제대로 작동하지 않음
# weight값이 계산됨 -> 이 w값으로 predict값을 예측함

# 4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("acc : ", acc)

y_predict = model.predict(x4)  # x_test값을 넣었을 때의 모델을 예측하라 / acc 외의 평가지표를 추가한 것
print(y_predict) # 11.0795, 12.024, 12.969.... 잘 구축된 모델임


# 만일 acc=0.0이면(epoch값을 1로 조정)? -0.93, -1.02, -1.1...

# 만일 model.evaluate에 x_train을 넣으면? y_predict = model.predict(x_train) / acc=0.1
# 1.64, 2.74, 3.84...
# 보통 훈련시키지 않은 test값을 넣는다

# 만일 model.evaluate에 x3을 넣으면? y_predict = model.predict(x3) / acc=0.5
# 95.73, 96.67, 97.62...

'''
이 코딩의 문제가 뭘까?
 -> x3에 매칭되는 y3의 값이 없다? 문제 없음 ex. 로또 번호 예측
 -> 데이터가 5개다? 문제 없음(코딩이 돌아감)

 => input_dim=1 (일차원 즉, 하나의 열) / 무조건 열 우선!
    x_train    x3
       1      101
       2      102
       3      103
       4      104
       5      105
       6
       7
       8
       9
      10

    input_dim=1 = input_shape=(1, ) => (?행, 1열)
    => x4에 20개의 데이터를 넣어보자 -> error! expected dense_1_input to have shape (1,) but got array with shape (20,)
    => ? x4 = np.array([range(30, 50)]) -> 1행 20열로 인식 -> dim=20 
    => 괄호 삭제 x4 = np.array(range(30, 50)) -> 20행 1열 -> dim=1 (행의 숫자는 상관 없음) -> 잘 돌아감

'''

