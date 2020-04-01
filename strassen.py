from __future__ import division
import math
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


#Helper Functions

#Given a matrix, returns a list of quadrants with the leftmost element being the quadrant in the first row and column of the matrix
#and the rightmost being the quadrant in the last row and column of the matrix.
def matrixIntoQuadrants(A):

    # the int rounds down
    length = len(A)
    size_of_quadrant = int(math.ceil(len(A)/2))

    quadrant1 = []
    quadrant2 = []
    quadrant3 = []
    quadrant4 = []
    for i in range(size_of_quadrant):
        quadrant1.append(A[i][:size_of_quadrant])
        quadrant2.append(A[i][size_of_quadrant:])
        if (i + size_of_quadrant) < length:
            quadrant3.append(A[i + size_of_quadrant][:size_of_quadrant])
            quadrant4.append(A[i + size_of_quadrant][size_of_quadrant:])
    return [quadrant1, quadrant2, quadrant3, quadrant4]

def matrix_add(A, B):
    length = len(A)
    matrix = []
    for i in range(length):
        matrix.append([])
        for j in range(length):
            matrix[i].append(A[i][j] + B[i][j])

    return matrix


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
        if i < len(C):
            C[i].extend(D[i])
            A.append(C[i])

    return A

# print(combineQuadrants([[1,2],[1,2]], [[3],[3]], [[1,2]], [[3]]))

#n is the desired matrix we want to achieve after padding
#this function is only functional for a certain n value with respect to the matrix provided.
def pad(M, n):
    length = len(M)

    if len(M) == n and len(M[0]) == n:
        return M

    elif len(M) == n:
        for i in range(length):
            M[i].append(0)
    elif len(M[0]) == n:
        M.append([0]* n)
    else:
        M.append([0]* length)
        length += 1
        for i in range(length):
            M[i].append(0)
    return M
            
def unpad(A, i):
    length = len(A)
    if i == 1:
        return A
    elif i == 2:
        for k in range(length):
            del A[k][-1]
    elif i == 3:
        del A[-1]
    else:
        for k in range(length):
            del A[k][-1]
        del A[-1]
    return A
# print(unpad([[1,0],[0,0]], 3))



#Given two matrices, multiply them using the strassen method. 
def strassen_mult(M, N):
    length = len(M)
    half = int(math.ceil(len(M)/2))
    padded = False
    if half != int(length/2):
        padded= True
    if length == 1:
        return [[M[0][0]*N[0][0]]]

    if length <= 97:
        return standard_mult(M,N)


    A, B, C, D = matrixIntoQuadrants(M)
    E, F, G, H = matrixIntoQuadrants(N)

    pad_A= pad(A, half) 
    pad_B = pad(B, half) 
    pad_C= pad(C, half) 
    pad_D= pad(D, half) 
    pad_E = pad(E, half) 
    pad_F = pad(F, half) 
    pad_G = pad(G, half) 
    pad_H = pad(H, half) 

    s1 = strassen_mult(pad_A, matrix_subtract(pad_F, pad_H))
    s2 = strassen_mult(matrix_add(pad_A, pad_B), pad_H)
    s3 = strassen_mult(matrix_add(pad_C, pad_D), pad_E)
    s4 = strassen_mult(pad_D, matrix_subtract(pad_G, pad_E))
    s5 = strassen_mult(matrix_add(pad_A, pad_D), matrix_add(pad_E, pad_H))
    s6 = strassen_mult(matrix_subtract(pad_B, pad_D), matrix_add(pad_G, pad_H))
    s7 = strassen_mult(matrix_subtract(pad_A, pad_C), matrix_add(pad_E, pad_F))

    matA = matrix_subtract(matrix_add(matrix_add(s5,s4), s6),s2)
    matB = matrix_add(s1,s2)
    matC = matrix_add(s3,s4)
    matD = matrix_subtract(matrix_subtract(matrix_add(s5,s1), s3),s7)



    if padded==True:
        matA = unpad(matA, 1)
        matB = unpad(matB, 2)
        matC = unpad(matC, 3)
        matD = unpad(matD, 4)

    result = combineQuadrants(matA, matB, matC, matD)
    # print(str(result) +"result")

    return result


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
            if i == k:
                matrix[i][k] = 0
            else:
                randElmt = weighted_choice(p)
                matrix[i][k] = randElmt
                matrix[k][i] = randElmt
        j+=1
    return matrix

# print(generateRandomMatrix(1024,0.2))

def numTriangles(p):
    A = generateRandomMatrix(1024,p)
    intermediateRes = strassen_mult(A, A)
    result = strassen_mult(A, intermediateRes)
    addDiagonals = 0
    for i in range(1024):
        addDiagonals += result[i][i]

    return addDiagonals/6

print(numTriangles(0.04))



# print(strassen_mult([[2,3],[3,4]], [[2,3],[3,4]]))
a = 194
L =  range(1,a+1)
A = [L[:] for _ in range(a)]
B = [L[:] for _ in range(a)]

# sumof = 0
# for i in range(30):
#     start = time.time()
#     standard_mult(A, B)
#     end = time.time()
#     r1 = end - start