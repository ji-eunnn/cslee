'''
X : (6, 3, 3) Y : (6, 3) 모델을 Y : (6, 1), (6, 1), (6, 1) 세 모델로 나누기
t26의 반대 경우

A = Input   <- (3, 3)
B = Conv1D(A)
C = MaxPooling1D(B)
D = Flatten(C)
/* Flatten 다음에 Dense층 추가해도 됨 */

모델1)
E = Dense(D)
F = Dense(1, )(E)

모델2)
G = Dense(D)
H = Dense(1, )(G)

모델3)
I = Dense(D)
J = Dense(1, )(I)

'''
# multivariate output 1d cnn example
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

# choose a number of time steps
n_steps = 3

# convert into input/output
X, y = split_sequence(dataset, n_steps)
print(X.shape, y.shape)  # (6, 3, 3) (6, 3)

# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]

# seperate output
y1 = y[:, 0].reshape((y.shape[0], 1))
y2 = y[:, 1].reshape((y.shape[0], 1))
y3 = y[:, 2].reshape((y.shape[0], 1))


from keras.layers import Dense, Input
from keras.models import Model

# define model
visible1 = Input(shape=(n_steps, n_features))
cnn1 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible1)
cnn1 = MaxPooling1D(pool_size=2)(cnn1)
cnn1 = Flatten()(cnn1)
cnn1 = Dense(50, activation='relu')(cnn1)
# define output 1
output1 = Dense(1, )(cnn1)
# define output 2
output2 = Dense(1, )(cnn1)
# define output 3
output3 = Dense(1, )(cnn1)

model = Model(inputs=[visible1], outputs=[output1, output2, output3])
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(X, [y1, y2, y3], epochs=100, verbose=1)

# demonstrate prediction
x_input = array([[70, 80, 90], [75, 85, 95], [145, 165, 185]])
x_input = x_input.reshape((1, n_steps, n_features))
# x_input = array([[70, 75, 145], [80, 85, 165], [90, 95, 185]])  # reshape 없이
yhat = model.predict(x_input, verbose=0)
print(yhat)
print(x_input)