def mul(A,B):
    """
    returns the product of the matrix A by the matrix B
    returns the empty list [] if the matrix dimensions
    are not consistent for multiplication
    """
    # check if the number of columns in A is equal to the number of rows in B
    if len(A[0]) != len(B):
        return []
    # create the result matrix
    C = []
    # for each row in A
    for i in range(len(A)):
        # create a new row in C
        C.append([])
        # for each column in B
        for j in range(len(B[0])):
            # compute the dot product of the i-th row in A and the j-th column in B
            C[i].append(sum([A[i][k]*B[k][j] for k in range(len(A[0]))]))
    return C

# function body
# test the function
A = [[1, 0, 0],
[0, 0, 3],
[0, 2, 0]]
B = [[1, 1],
[0, .5],
[2, 1/3.0]]
C = [[ 1, 0, 0 ],
[ 0, 0, 0.5],
[ 0, 1/3.0, 0]]

print(mul(A,B))
print(mul(B,A))
print(mul(A,C))
