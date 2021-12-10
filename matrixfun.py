#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kavi Freyaldenhoven
"""

"""
printMat

This function will print a matrix in a readable format. You will not need to
alter this function.

INPUTS
mat: the matrix represented as a list of lists in row major form.

OUTPUTS
s: a string representing the matrix formatted nicely.
"""
def printMat(mat):
    s = ''
    for row in mat:
        s = s + ''
        for col in row:
            # Display 2 digits after the decimal, using 9 chars.
            s = s + ('%9.2e' % col) + ' '
        s = s + '\n'
    return s

"""
Matrix Class

This class will have code that implements:
- matrix multiplication
- LU factorization
- backward substitution
- forward substitution
- permutation
- Gaussian elimination
"""
class Matrix:

    """
    Class attributes:
    mat:     the matrix itself, represented as a list of lists.
    numRows: the number of rows in the matrix.
    numCols: the number of columns in the matrix.
    L:       the lower triangular matrix from the LU Factorization.
    U:       the upper triangular matrix from the LU Factorization.
    P:       the permutation matrix from the LU Factorization.
    ipiv:    the permutation vector from the LU Factorization.
    """

    # Constructor method.
    def __init__(self, mat):
        self.mat = mat
        self.numRows = len(mat)
        self.numCols = len(mat[0])
        self.L = None
        self.U = None
        self.P = None
        self.ipiv = None

    # Special method used for printing this Matrix.

    def __repr__(self):
        s = ''
        s += 'The %dx%d Matrix itself:\n\n' % (self.numRows, self.numCols)
        s += printMat(self.mat)
        s += '\n'
        if self.L != None:
            s += 'The lower triangular matrix L:\n\n'
            s += printMat(self.L.mat)
            s += '\n'
        if self.U != None:
            s += 'The upper triangular matrix U:\n\n'
            s += printMat(self.U.mat)
            s += '\n'
        if self.P != None:
            s += 'The permutation matrix P:\n\n'
            s += printMat(self.P.mat)
            s += '\n'
        if self.ipiv != None:
            s += 'The permutation vector ipiv:\n\n'
            s += printMat([self.ipiv])
            s += '\n'
        return s
    
    ''' matMult
    
    This method performs matrix multiplication of self * B and 
    then returns the result
    
    Inputs: B is a matrix of class matrix
    
    Output: C is a matrix of class matrix that is the product of self and B
    
    '''    
    
    def matMult(self,B):
        
       # check to make sure dimensions match
        
        if self.numCols != B.numRows:
            raise ValueError('Matrix Dimension Mismatch!')
    
        # initialize new matrix D using a list comprehension
        
        C = Matrix([[0.0 for row in range(B.numCols)] for col in range(self.numCols)])
        
        # for loops for multiplication
        
        for i in range(self.numRows):
            for j in range(B.numCols):
                for k in range(self.numCols):
                    C.mat[i][j] += self.mat[i][k] * B.mat[k][j]
                    
        return C 
    
    ''' LUfact
    
    This method performs the LU Factorization of the matrix self using
    partial pivoting. It then stores the results. 
    
    Inputs: There are no inputs
    
    Outputs: There are no outputs
    
    '''

    def LUfact(self):
        
        A = self.mat
        
        # create a copy for A
        U = [ A[ind].copy() for ind in range(len(A)) ]
        
        # initialize matrix L and permutation matrix P
        
        P = [ [0.0]*j + [1] + [0]*(len(A)-1-j) for j in range(len(A)) ]
        L = [ [0.0]*j + [1] + [0]*(len(A)-1-j) for j in range(len(A)) ]
        ipiv = [j for j in range(len(A))]
        
        # implementing partial pivoting
        for j in range(0, len(U)-1):
            pivotRow = j
            for i in range(j, len(U)):
                if abs(U[i][j]) > abs(U[pivotRow][j]):
                    pivotRow = i
                    
            # making sure matrix is invertible       
            if U[pivotRow][j] == 0:
                raise ValueError('Matrix is singular')
                
            # Swapping rows in the matrices
            tempRow = U[j]
            U[j] = U[pivotRow]
            U[pivotRow] = tempRow
            
            tempRow = P[j]
            P[j] = P[pivotRow]
            P[pivotRow] = tempRow
            
            tempRow = ipiv[j]
            ipiv[j] = ipiv[pivotRow]
            ipiv[pivotRow] = tempRow
                
            
            # entries for matrix L
            for i in range(0,j):
                tempValue = L[j][i]
                L[j][i] = L[pivotRow][i]
                L[pivotRow][i] = tempValue
        
            # Finishing LU factorization with elimination step
            for k in range(j+1,len(U)):
                l = U[k][j]/U[j][j]
                L[k][j] = l
                for c in range(j,len(U)):
                    U[k][c] -= l*U[j][c]
                    
        # final check for invertibility           
        if U[len(U[0])-1][len(U[0])-1] == 0:
            raise ValueError('Matrix is Singular')
                    
        # storing results            
        self.U = Matrix(U)
        self.L = Matrix(L)
        self.ipiv = ipiv
        self.P = Matrix(P)
        
        return
    
    ''' backSub
    
    Performs back substitution to solve the system self.U * x = c 
    given an input vector c
    
    Inputs: c is a vector of class matrix
    
    Output: x is a vector of class matrix
    
    '''

    def backSub(self,c):
        A = self.mat
        U = self.U.mat
        
        # initializing matrix x 
        x = [[0.0] for i in range(len(A))]
        
        # setting initial x sub n equal to it's correct value
        x[len(A)-1][0] = c.mat[len(A)-1][0] / U[len(A)-1][len(A)-1]
        
        # looping backwards through the unknowns
        for i in range(len(A)-1):
            sum = 0 
            j = len(A) - 2 - i 
            for k in range (j+1, len(A)):
                sum += U[j][k] * x[k][0]
            x[j][0] = (1 / U[j][j]) * (c.mat[j][0] - sum)
        
        return Matrix(x)
    
    ''' forSub
    
    Performs forward substitution to solve the system self.L * c = b 
    given an input vector c
    
    Inputs: b is a vector of class matrix
    
    Output: a vector of class matrix
    
    '''    

    def forSub(self,b):
        A = self.mat
        L = self.L.mat
        
        # initializing matrix c
        c = [[0.0] for i in range(len(A))]
        
        # looping forwards through the unknowns
        for i in range(len(A)):
            sum = 0
            for j in range(i):
                sum += L[i][j] * c[j][0]
            c[i][0] = (b.mat[i][0] - sum) / L[i][i]
            
        return Matrix(c)
    
    ''' permute
    
    permutes the input vector b according to the permutations stored in the 
    matrix P 
    
    Inputs: b is a vector of class matrix
    
    Output: bHat is a vector of class matrix
    
    '''    

    def permute(self,b):
        # creating bHat
        bHat = self.P.matMult(b)
        return bHat
    
    ''' gaussElim
    
    Performs Gaussian Elimination to solve the system self * x = b
    given an input vector b
    
    Inputs: b is a vector of class matrix
    
    Output: x is a vector of matrix class
    
    '''

    def gaussElim(self,b):
        
        A = self
        
        # performing LU factorization if it hasn't been done
        if A.U == None:
            A.LUfact()
        
        # obtaining permutation vector and then solving the system
        bHat = A.P.matMult(b)
        c = A.forSub(bHat)
        x = A.backSub(c)
        
        return x   
