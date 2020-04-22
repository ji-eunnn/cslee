# 앙상블 : 변수를 조합

'''
변수가 두 개 이상인 경우 ex. 타이타닉 - 성별, 등급, 직업...
xxx1, xxx2, xxx3처럼 변수를 3개로 잡으면 모델을 돌리기가 애매(세 개의 모델을 만들어서 각각 돌릴 것인가?) -> 변수는 오직 하나여야 함 xxx
데이터를 하나로 합치는게 아니라 두 개의 데이터로부터 두 모델을 생성하고 합친다
'''

# 1. 데이터 구성
import numpy as np
from sklearn.model_selection import train_test_split

xxx1 = np.array([range(100), range(311, 411), range(201, 301)])
xxx2 = np.array([range(100), range(311, 411), range(201, 301)])
yyy1 = np.array([range(501, 601), range(611, 711), range(101,201)])
yyy2 = np.array([range(501, 601), range(611, 711), range(101,201)])

xxx1 = np.transpose(xxx1) 
xxx2 = np.transpose(xxx2) 
yyy1 = np.transpose(yyy1) 
yyy2 = np.transpose(yyy2) # dimention = 3    input_shape = (3, )

x1_train, x1_test, y1_train, y1_test = train_test_split(xxx1, yyy1, test_size=0.2, random_state=66)
x1_train, x1_val, y1_train, y1_val = train_test_split(x1_train, y1_train, test_size=0.25, random_state=66)

x2_train, x2_test, y2_train, y2_test = train_test_split(xxx2, yyy2, test_size=0.2, random_state=66)
x2_train, x2_val, y2_train, y2_val = train_test_split(x2_train, y2_train, test_size=0.25, random_state=66)


# 2. 모델 구성
from keras.layers import Dense, Input, Concatenate
from keras.models import Sequential, Model
from keras.layers.merge import concatenate

# model 1
input1 = Input(shape=(3,))
dense1 = Dense(150, activation='relu')(input1)  # 모델 1개
# model 2
input2 = Input(shape=(3,))
dense2 = Dense(60, activation='relu')(input1)
dense21 = Dense(30, activation='relu')(dense2)  # 모델 2개
# merge
merge1 = concatenate([dense1, dense21])  

output_11 = Dense(80)(merge1)
output_12 = Dense(10)(output_11)
merge2 = Dense(2)(output_12)    # 모델 3개

output_1 = Dense(30)(merge2)
output1 = Dense(3)(output_1)    # 모델 4개
output_2 = Dense(70)(merge2)
output2 = Dense(3)(output_2)    # 모델 5개

model = Model(inputs=[input1, input2], outputs=[output1, output2])

model.summary()


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=100, batch_size=1, validation_data=([x1_val, x2_val], [y1_val, y2_val]))

# 4. 평가 예측
# loss, acc = model.evaluate(x_test, y_test, batch_size=1)
acc = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)  # 앞의 세 개만.. but 회귀 지표가 아닌 분류 지표 -> 신경X
print("acc : ", acc)
# print("loss : ", loss)

y1_predict, y2_predict = model.predict([x1_test, x2_test])
print(y1_predict)
print(y2_predict)   # test값과 비교


from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE평균 : ", (RMSE1 + RMSE2)/2)  # RMSE는 낮을수록 좋다

from sklearn.metrics import r2_score 
r2_y1_predict = r2_score(y1_test, y1_predict)  
print("R2_1 : ", r2_y1_predict)
r2_y2_predict = r2_score(y2_test, y2_predict)  
print("R2_2 : ", r2_y2_predict)
print("R2 : ", (r2_y1_predict + r2_y2_predict)/2)



# 여기까지가 DNN(NN)


