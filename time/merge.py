'''
sequential() 모델 두 개 만들어서 merge -> 하나의 sequential 모델로? X   merge 뒤에는 sequential 모델 사용할 수 없음, 함수형 모델로!

'''

import numpy as np
x1 = np.array([1,2,3])
y1 = np.array([1,2,3])

x2 = np.array([3,4,5])
y2 = np.array([3,4,5])

from keras.layers import Dense, Input, Concatenate
from keras.models import Sequential, Model
from keras.layers.merge import concatenate

# model1
model1 = Sequential()
model1.add(Dense(5, input_dim=1, activation='relu'))
model1.add(Dense(3))
model1.add(Dense(4))
model1.add(Dense(1))

model1.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model1.fit(x1, y1, epochs=100)

model1.summary()


# model2
model2 = Sequential()
model2.add(Dense(5, input_dim=1, activation='relu'))  
model2.add(Dense(3))
model2.add(Dense(4))
model2.add(Dense(1))

model2.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model2.fit(x2, y2, epochs=100)

model2.summary()


# merge
merge = concatenate([model1, ])  