import numpy as np

A = np.random.rand(200,10)

mu = np.zeros(A.shape[1])
for i in range(A.shape[0]):
    mu += A[i]
mu /= A.shape[0]
# mu is the mean of the columns of A

B = np.zeros_like(A)
for i in range(A.shape[0]):
    B[i] = A[i] - mu
# B is the matrix A with the mean of the columns subtracted from each row

# or in a single line
B = A - A.mean(axis=0)