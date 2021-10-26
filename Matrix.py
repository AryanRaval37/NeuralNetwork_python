import random
import numbers
from typing import List


class Matrix:
    def __init__(self, rows, cols, name=None):
        self.rows = rows
        self.cols = cols
        self.name = name
        self.data = [[0 for i in range(self.cols)] for j in range(self.rows)]

    def __str__(self):
        print("\n")
        for i in range(self.rows):
            # print("\t")
            for j in range(self.cols):
                print(self.data[i][j], end="\t")
            print("\n")
        if self.name is None:
            return "Matrix : \n" + f"\tRows: {self.rows}\n" + f"\tCols: {self.cols}\n"
        else:
            return (
                f"Matrix : {self.name}\n"
                + f"\tRows: {self.rows}\n"
                + f"\tCols: {self.cols}\n"
            )

    def multiply(self, n):
        if isinstance(n, Matrix):
            assert (
                self.rows == n.rows and self.cols == n.cols
            ), "Invalid Matrix Provided"
            self.data = [
                [(self.data[i][j] * n.data[i][j]) for j in range(self.cols)]
                for i in range(self.rows)
            ]

    def multiply(m1, m2):
        assert m1.cols == m2.rows, "Cols of m1 are not equal to rows of m2"
        if m1.name is None or m2.name is None:
            result = Matrix(m1.rows, m2.cols)
        else:
            result = Matrix(m1.rows, m2.cols, f"Dot product ({m1.name}.{m2.name})")
        for i in range(result.rows):
            for j in range(result.cols):
                sum = 0
                for k in range(m1.cols):
                    sum += m1.data[i][k] * m2.data[k][j]
                result.data[i][j] = sum
        return result

    def toList(self):
        arr = []
        for i in range(self.rows):
            for j in range(self.cols):
                arr.append(self.data[i][j])
        return arr

    def map(self, fn):
        self.data = [
            [(fn(self.data[i][j])) for j in range(self.cols)] for i in range(self.rows)
        ]

    def map(m, fn):
        m.name = f"{m.name} (Mapped)"
        m.data = [[fn(m.data[i][j]) for j in range(m.cols)] for i in range(m.rows)]
        return m

    def toMatrix(a, name=None):
        assert isinstance(a, List), "Invalid Parameters."
        if name is None:
            m = Matrix(len(a), 1)
        else:
            m = Matrix(len(a), 1, name)
        for i in range(len(a)):
            m.data[i][0] = a[i]
        return m

    def randomize(self):
        self.data = [
            [random.random() for i in range(self.cols)] for j in range(self.rows)
        ]

    def subtract(a, b):
        assert isinstance(a, Matrix) and isinstance(b, Matrix), "Invalid Parameters."
        assert a.rows == b.rows and a.cols == b.cols, "Invalid Parameters"
        if a.name is None or b.name is None:
            result = Matrix(a.rows, a.cols, "Results")
        else:
            result = Matrix(a.rows, a.cols, f"Results({a.name}-{b.name})")
        result.data = [
            [(a.data[i][j] - b.data[i][j]) for j in range(a.cols)]
            for i in range(a.rows)
        ]
        return result

    def add(self, n):
        assert isinstance(n, Matrix) or isinstance(
            n, numbers.Number
        ), "\n\n\nInvalid Parameters\n"
        if isinstance(n, Matrix):
            assert n.rows == self.rows and n.cols == self.cols, "Invalid Parameters"
            self.data = [
                [(self.data[i][j] + n.data[i][j]) for j in range(self.cols)]
                for i in range(self.rows)
            ]
        else:
            self.data = [
                [(self.data[i][j] + n) for j in range(self.cols)]
                for i in range(self.rows)
            ]

    def transpose(m):
        if m.name is None:
            result = Matrix(m.cols, m.rows, "Result")
        else:
            result = Matrix(m.cols, m.rows, f"{m.name} (Transposed)")
        for i in range(m.rows):
            for j in range(m.cols):
                result.data[j][i] = m.data[i][j]
        return result
