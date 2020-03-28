

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

# print(standard_mult([[1,2,3], [4,5,6], [7,8,9]], [[2,2,2],[2,1,2],[1,1,1]] ))


#Helper Functions

#Given a matrix, returns a list of quadrants with the leftmost element being the quadrant in the first row and column of the matrix
#and the rightmost being the quadrant in the last row and column of the matrix.
def matrixIntoQuadrants(A):
    size_of_quadrant = len(A)/2
    quadrant1 = []
    quadrant2 = []
    quadrant3 = []
    quadrant4 = []
    for i in range(size_of_quadrant):
        quadrant1.append(A[i][:size_of_quadrant])
        quadrant2.append(A[i][size_of_quadrant:])
        quadrant3.append(A[i + size_of_quadrant][:size_of_quadrant])
        quadrant4.append(A[i + size_of_quadrant][size_of_quadrant:])
    return [quadrant1, quadrant2, quadrant3, quadrant4]
# print(matrixIntoQuadrants([[2,3, 1,2], [1,2,2,1], [5,4,1,2], [1,2,3,1]]))
        

def matrix_add(A, B):
    length = len(A)
    matrix = []
    for i in range(length):
        matrix.append([])
        for j in range(length):
            matrix[i].append(A[i][j] + B[i][j])

    return matrix

# print(matrix_add([[2,3, 1,2], [1,2,2,1], [5,4,1,2], [1,2,3,1]], [[2,3, 1,2], [1,2,2,1], [5,4,1,2], [1,2,3,1]]))

def matrix_subtract(A, B):
    length = len(A)
    matrix = []
    for i in range(length):
        matrix.append([])
        for j in range(length):
            matrix[i].append(A[i][j] - B[i][j])

    return matrix


#Given two matrices, multiply them using the strassen method. 
def strassen_mult(M, N):
    if len(M) == 1:
        return M[0][0]*N[0][0]

    A, B, C, D = matrixIntoQuadrants(M)
    E, F, G, H = matrixIntoQuadrants(N)
    
    s1 = strassen_mult(A, matrix_subtract(F, H))
    s2 = strassen_mult(matrix_add(A, B), H)
    s3 = strassen_mult(matrix_add(C, D), E)
    s4 = strassen_mult(D, matrix_subtract(G, E))
    s5 = strassen_mult(matrix_add(A, D), matrix_add(E, H))
    s6 = strassen_mult(matrix_subtract(B, D), matrix_add(G, H))
    s7 = strassen_mult(matrix_subtract(A, C), matrix_add(E, F))

    return [[s5+s4-s2+s6, s1+s2], [s3+s4, s5+s1-s3-s7]]
 

print(strassen_mult([[2,3],[3,4]], [[2,3],[3,4]]))
    



