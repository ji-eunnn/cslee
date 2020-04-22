# input이 list가 아닌 행렬인 경우!
# input 6, 3, 3 -> output 6, 3
from numpy import array, hstack

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
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])  # (1, 9)
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))  # (9, 1)
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

# horizontally stack columns : 수직(h)으로 데이터 붙이기
dataset = hstack((in_seq1, in_seq2, out_seq))  # (9, 3)

# choose a number of time steps
n_steps = 3

# split into samples
X, y = split_sequence(dataset, n_steps)
# print(X.shape, y.shape)

# summarize the data
for i in range(len(X)):
    print(X[i], y[i])