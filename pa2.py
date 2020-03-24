

#Given two n*n matrices, multiply them using the standard method. Assume matrices are represented as a list of lists.
def standard_mult(A, B):
    n = len(A)
    resulting_matrix = []
    for i in range(n):
        resulting_matrix.append([])
        for j in range(n):
            value = 0
            for k in range(n):
                value += A[i][k] * B[k][j]
            resulting_matrix[i].append(value)
    return resulting_matrix

print(standard_mult([[1,2,3], [4,5,6], [7,8,9]], [[2,2,2],[2,1,2],[1,1,1]] ))

def strassen_mult(A, B):
    pass