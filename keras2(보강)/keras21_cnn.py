from keras.models import Sequential

filter_size = 32
kernel_size = (3, 3)

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
model = Sequential()
model.add(Conv2D(7, (2, 2), padding='same', # same, valid 디폴트    # output : 7, 이미지를 2x2로 자르겠다
                input_shape=(10,10,1)))     # 10*10짜리 이미지를 2*2로 자르면? -> 9*9*7(9*9짜리 이미지가 7장) but padding='same'이니까 -> 10*10*7
model.add(Conv2D(16, (2, 2)))   # -> 9*9*16이 됨
model.add(MaxPooling2D(3,3))    # -> 3*3*16
model.add(Conv2D(8, (2, 2)))    # -> 2*2*8 
model.add(Flatten())
model.add(Dense(1))
model.summary()

'''
4x4 이미지를 2x2로 자르면 -> 3x3
'''