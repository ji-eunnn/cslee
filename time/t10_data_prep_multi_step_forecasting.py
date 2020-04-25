'''
split_sequence()

1
2
3
2
5
6
7
5 (오늘 주가)
---
? (내일 주가)

5개씩 split =>
    X       Y
1 2 3 2 5 | 6
2 3 2 5 6 | 7
3 2 5 6 7 | 5
-------------
2 5 6 7 5 | ?  <- predict


'''


# multi-step data preparation
from numpy import array

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

'''
     X        Y
[10 20 30] [40 50]
[20 30 40] [50 60]
[30 40 50] [60 70]
[40 50 60] [70 80]
[50 60 70] [80 90]

'''

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]  # (9, ) reshape 필요
# choose a number of time steps(X를 3개씩, Y를 2개씩 자르자)
n_steps_in, n_steps_out = 3, 2  
# split into samples
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
# summarize the data
for i in range(len(X)) :
    print(X[i], y[i])

print(X.shape, y.shape) # (5, 3) (5, 2)