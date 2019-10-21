#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Matrix():
    def __init__(self, data):
        if type(data) == int:
            self.data = [[float(i == j) for i in range(data)] for j in range(data)]
        else:
            self.data = [[float(c) for c in r] for r in data]

    def row(self):
        return len(self.data)

    def col(self):
        return len(self.data[0])

    def show(self):
        for i in range(self.row()):
            print(self.data[i])

    def transpose(self):
        return Matrix([[self.data[i][j] for i in range(self.row())]
                                        for j in range(self.col())])

    def row_op(self, r1, r2, n):
        for j in range(self.col()):
            self.data[r1][j] += self.data[r2][j] * n

    def LU(self):
        L = Matrix(self.row())
        U = self.copy()
        for j in range(0, self.col() - 1):
            for i in range(j + 1, self.row()):
                L.data[i][j] = U.data[i][j] / U.data[j][j]
                U.row_op(i, j, -L.data[i][j])
        return L, U

    def inverse(self):
        L, U = self.LU()
        Y = Matrix(L.row())
        for j in range(Y.col()):
            for i in range(L.row()):
                s = float(i == j) - sum([L.data[i][k] * Y.data[k][j] for k in range(i)])
                Y.data[i][j] = s / L.data[i][i]

        A_in = Matrix(U.row())
        for j in range(A_in.col()-1, -1, -1):
            for i in range(U.row()-1, -1, -1):
                s = Y.data[i][j] - sum([U.data[i][k] * A_in.data[k][j] for k in range(U.row()-1, i, -1)])
                A_in.data[i][j] = s / U.data[i][i]
        return A_in


    def copy(self):
        return self * 1

    def __add__(self, matrix):
        return Matrix([[self.data[i][j] + matrix.data[i][j] for j in range(self.col())]
                                                            for i in range(self.row())])

    def __sub__(self, matrix):
        return self + matrix * -1

    def __mul__(self, matrix):
        if type(matrix) == int or type(matrix) == float:
            return Matrix([[self.data[i][j] * matrix for j in range(self.col())]
                                                     for i in range(self.row())])
        else:
            return Matrix([[sum([self.data[i][k] * matrix.data[k][j] for k in range(self.col())])
                                                                     for j in range(matrix.col())]
                                                                     for i in range(self.row())])