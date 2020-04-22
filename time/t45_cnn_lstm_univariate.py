'''
시분할 - Conv1D를 시분할 해보자
TimeDistributed <- wrapper
'''
# univariate data preparation
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten, TimeDistributed
from keras.layers.convolutional import Conv1D, MaxPooling1D


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) -1 :
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# choose a number of time steps
n_steps = 4

# split into samples
X, y = split_sequence(raw_seq, n_steps)
print(X.shape)  # (5, 4)

# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
n_features = 1
n_seq = 2
n_steps = 2
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
print(X.shape)  # (5, 2, 2, 1)

# define model
model = Sequential()
# 4차원
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),
                                input_shape=(100, n_steps, n_features)))  # 2, 2, 1  # None 대신 1, 2, ..., 100을 넣어보자 -> 돌아감!(2, 2, 1짜리를 100으로 랩핑 했는데도!)
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

'''
# fit model
model.fit(X, y, epochs=500, verbose=0)

# demonstrate prediction
x_input = array([60, 70, 80, 90])
x_input = x_input.reshape((1, n_seq, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

'''