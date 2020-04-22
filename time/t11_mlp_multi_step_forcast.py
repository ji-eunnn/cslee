'''
model 추가
'''

# multi-step data preparation
from numpy import array
from keras.models import Sequential
from keras.layers import Dense

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]  # (9, ) reshape 필요

# choose a number of time steps(X를 3개씩, Y를 2개씩 자르자)
n_steps_in, n_steps_out = 3, 2  

# split into samples
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)

# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_steps_in))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(X, y, epochs=2000, verbose=0)

# demonstrate prediction
x_input = array([70, 80, 90]) # 1행 3열로 보이지만 벡터가 3개 => (3, ) => reshape 필요
x_input = x_input.reshape(1, 3)
y_pred = model.predict(x_input)
print(y_pred)


# summarize the data
for i in range(len(X)) :
    print(X[i], y[i])

print(X.shape, y.shape) # (5, 3) (5, 2) : 행은 무시