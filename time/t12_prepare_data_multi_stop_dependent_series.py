'''
행렬일 때(주가, 금, 채권, ....)
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
        out_end_ix = end_ix + n_steps_out -1
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix, :-1], sequence[end_ix-1:out_end_ix, -1]
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

# split into samples
X, y = split_sequence(dataset, n_steps_in, n_steps_out)
print(X.shape, y.shape)  # (6, 3, 2) (6, 2) : 행은 무시

# summarize the data
for i in range(len(X)) :
    print(X[i], y[i])



'''
삼성 주가  LG 주가 | 종합 주가 (알고싶은 건 이틀치)
---------------
| 10         15 |      25
| 20         25 |      45
                      ----
| 30         35 |    | 65 |
---------------
  40         45      | 85 |
                      ---- 
  50         55       105
  60         65       125
  70         75       145
  80         85       165
  90         95       185



'''