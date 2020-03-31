import time
import random

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
small =  [[0,1,1], [1,0,1],[1,1,0]]
res1 = standard_mult(small, small)
# print(standard_mult(res1, small))
A = [[0,1,1,1], [1,0,1,0],[1,1,0,1],[1,0,1,0]]
res = standard_mult(A, A)
# print(standard_mult(res, A))

#Helper Functions

#Given a matrix, returns a list of quadrants with the leftmost element being the quadrant in the first row and column of the matrix
#and the rightmost being the quadrant in the last row and column of the matrix.
def matrixIntoQuadrants(A):

    # the int rounds down
    size_of_quadrant = int(len(A)/2)

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


def combineQuadrants(A, B, C, D):
    length = len(A)
    for i in range(length):
        A[i].extend(B[i])
        C[i].extend(D[i])
        A.append(C[i])

    return A

# print(combineQuadrants([[1,2],[3,4]], [[5,6],[7,8]], [[9,10],[11,12]], [[13,14],[15,16]]))


#Given two matrices, multiply them using the strassen method. 
def strassen_mult(M, N):
    if len(M) == 1:
        return [[M[0][0]*N[0][0]]]

    A, B, C, D = matrixIntoQuadrants(M)
    E, F, G, H = matrixIntoQuadrants(N)
    
    s1 = strassen_mult(A, matrix_subtract(F, H))
    s2 = strassen_mult(matrix_add(A, B), H)
    s3 = strassen_mult(matrix_add(C, D), E)
    s4 = strassen_mult(D, matrix_subtract(G, E))
    s5 = strassen_mult(matrix_add(A, D), matrix_add(E, H))
    s6 = strassen_mult(matrix_subtract(B, D), matrix_add(G, H))
    s7 = strassen_mult(matrix_subtract(A, C), matrix_add(E, F))

    matA = matrix_subtract(matrix_add(matrix_add(s5,s4), s6),s2)
    matB = matrix_add(s1,s2)
    matC = matrix_add(s3,s4)
    matD = matrix_subtract(matrix_subtract(matrix_add(s5,s1), s3),s7)

    return(combineQuadrants(matA, matB, matC, matD))



def weighted_choice(p):
    probs = [1-p, p]
    r = random.random()
    index = 0
    while(r >= 0 and index < len(probs)):
      r -= probs[index]
      index += 1
    return index - 1
# print(weighted_choice(0.5))


def generateRandomMatrix(n, p):
    seq = [0,1]
    weights = [p-1, p]
    matrix = []
    for i in range(n):
        matrix.append([0]*n)
    j = 1
    for i in range(n):
        for k in range(j):
            randElmt = weighted_choice(0.5)
            matrix[i][k] = randElmt
            matrix[k][i] = randElmt
        j+=1
    return matrix

# print(generateRandomMatrix(1024,0.2))

def numTriangles(p):
    A = generateRandomMatrix(1024,p)
    intermediateRes = standard_mult(A, A)
    result = standard_mult(A, intermediateRes)
    addDiagonals = 0
    for i in range(1024):
        addDiagonals += result[i][i]

    return addDiagonals/6

print(numTriangles(0.01))



# print(strassen_mult([[2,3],[3,4]], [[2,3],[3,4]]))
a = 1024
L =  range(1,a+1)
A = [L]*(a)


start = time.time()
# print(standard_mult(A, A)[0])
end = time.time()
# print(end - start)

timeMult = 0.0252149105072

timeStrassen = 0.67314696312

result = timeMult - timeStrassen
# print(result)






















