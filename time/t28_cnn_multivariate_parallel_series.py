'''
y의 디멘션이 3일 때
'''

from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# convert to [rows, columns]
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# print(dataset)

# choose a number of time steps
n_steps = 3

# convert into input/output
X, y = split_sequence(dataset, n_steps)
print(X.shape, y.shape)  # (6, 3, 3) (6, 3)

# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]

# define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features))) 
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten()) 
model.add(Dense(50, activation='relu'))
model.add(Dense(100))
model.add(Dense(25))
model.add(Dense(3)) # 3!
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(X, y, epochs=100, verbose=1)

# demonstrate prediction
x_input = array([[70, 80, 90], [75, 85, 95], [145, 165, 185]])
x_input = x_input.reshape((1, n_steps, n_features))
# x_input = array([[70, 75, 145], [80, 85, 165], [90, 95, 185]])  # reshape 없이
yhat = model.predict(x_input, verbose=0)
print(yhat)
print(x_input)