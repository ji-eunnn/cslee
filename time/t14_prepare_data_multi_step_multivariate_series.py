'''
(X.shape, y.shape) -> (5, 3, 3) (5, 2, 3) 인 경우
X도 Y도 flatten 해주어야 한다.
X -> 3*3=9
Y -> 2*3=6

+
Dense층 input_dim, output_dim 수정

'''

# multi-step data preparation
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from numpy import hstack

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
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])  # (9, )
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])

# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))

# choose a number of time steps(X를 3개씩, Y를 2개씩 자르자)
n_steps_in, n_steps_out = 3, 2  

# convert into input/output
X, y = split_sequence(dataset, n_steps_in, n_steps_out)
print(X.shape, y.shape)  # (5, 3, 3) (5, 2, 3)

# flatten input
n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))  # (5, 9)
n_output = y.shape[1] * y.shape[2]
y = y.reshape((y.shape[0], n_output))

# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_input))
model.add(Dense(n_output))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(X, y, epochs=2000, verbose=0)

# demonstrate prediction
x_input = array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])
x_input = x_input.reshape((1, n_input))
yhat = model.predict(x_input, verbose=0)
print(yhat)

# summarize the data
for i in range(len(X)) :
    print(X[i], y[i])
