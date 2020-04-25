from keras.models import Sequential
from keras.layers import Conv2D, Flatten

# filter_size = 32
# kernel_size = (3,3)
model = Sequential()

model.add(Conv2D(32, (2,2), padding = 'valid',
                input_shape = (7, 7, 1))) # 28*28 짜리 한 장

'''
model.add(Conv2D(filter_size, kernel_size, #padding = 'valid',
                input_shape = (28, 28, 1)))
filter_size : output, 임의로 주는 것
kernel_size : 3*3로 자르기, 임의로 주는 것
input_shape : 28*28 짜리 그림 한 장

'''
model.add(Conv2D(16, (3,3)))
model.add(Conv2D(100, (3,3)))


# from keras.layers import MaxPooling2D
# pool_size = (2, 2)
# model.add(MaxPooling2D(pool_size))
from keras.layers import MaxPooling2D
model.add(MaxPooling2D(pool_size=2))    # 각 조각에서 가장 큰 값만, 즉 특성 있는 값
model.add(Flatten())    # 그림을 조각낸 다음, 쫙 펴다

model.summary()
'''
Conv2D(32, (2,2), #padding = 'valid', input_shape = (7, 7, 1)) 결과
    output shape : (None, 6, 6, 32) 6*6짜리 32장 -> 데이터가 늘어남으로써 특성 잡기가 쉬어짐
Conv2D(16, (3,3)) 결과
    output shape : (None, 4, 4, 16) 
add(Conv2D(100, (3,3))) 결과
    output shape : (None, 2, 2, 100)   
model.add(MaxPooling2D(pool_size=2)) 결과
    output shape : (None, 1, 1, 100)
'''

'''
