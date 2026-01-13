import numpy as np
import numbers

class Matrix:
    def __init__(self, rows, cols, name=None, wait=False):
        self.rows = rows
        self.cols = cols
        self.name = name
        if wait:
            self.data = []
        else:
            np.random.uniform2 = lambda *args, dtype=np.float64: np.random.uniform(
                *args
            ).astype(dtype)
            self.data = np.random.uniform2(
                -0.5, 0.5, (self.rows, self.cols), dtype=np.float32
            )

    def __str__(self):
        print("\n")
        print(self.data)
        if self.name is None:
            return "Matrix : \n" + f"\tRows: {self.rows}\n" + f"\tCols: {self.cols}\n"
        else:
            return (
                f"Matrix : {self.name}\n"
                + f"\tRows: {self.rows}\n"
                + f"\tCols: {self.cols}\n"
            )

    def mean(self):
        return np.average(self.data)

    def simpleMultiply(self, n):
        if isinstance(n, Matrix):
            assert (
                self.rows == n.rows and self.cols == n.cols
            ), "Invalid Matrix Provided"
            self.data = np.multiply(self.data, n.data)
        elif isinstance(n, numbers.Number):
            self.data = self.data * np.float32(n)

    @staticmethod
    def multiply(m1, m2):
        assert (
            m1.cols == m2.rows
        ), f"Cols of m1 are not equal to rows of m2\nCols of m1 are {m1.cols}\nRows of m2 are {m2.rows}"
        if m1.name is None or m2.name is None:
            result = Matrix(m1.rows, m2.cols)
        else:
            result = Matrix(m1.rows, m2.cols, f"Dot product ({m1.name}.{m2.name})")
        result.data = np.dot(m1.data, m2.data)
        return result

    def toList(self):
        return self.data.flatten().tolist()

    def justFlatten(self):
        return self.data.flatten()

    def map(self, fn):
        self.data = fn(self.data)

    @staticmethod
    def map_static(m, fn):
        result = Matrix(m.rows, m.cols, f"{m.name} (Mapped)")
        result.data = fn(m.data)
        return result

    @staticmethod
    def toMatrix(a, name=None):
        assert isinstance(a, list), "Invalid Parameters."
        if name is None:
            m = Matrix(len(a), 1)
        else:
            m = Matrix(len(a), 1, name)
        m.data = np.array(a).reshape(len(a), 1)
        return m

    @staticmethod
    def subtract(a, b):
        assert isinstance(a, Matrix) and isinstance(b, Matrix), "Invalid Parameters."
        assert a.rows == b.rows and a.cols == b.cols, "Invalid Parameters"
        if a.name is None or b.name is None:
            result = Matrix(a.rows, a.cols, "Results")
        else:
            result = Matrix(a.rows, a.cols, f"Results({a.name}-{b.name})")
        result.data = a.data - b.data
        return result

    def add(self, n):
        assert isinstance(n, Matrix) or isinstance(
            n, numbers.Number
        ), "\n\n\nInvalid Parameters\n"
        if isinstance(n, Matrix):
            # Allow broadcasting if cols doesn't match but rows do (e.g. adding bias (R, 1) to (R, C))
            if n.rows == self.rows and n.cols == 1 and self.cols > 1:
                # Broadcasting n over self
                self.data = self.data + n.data
            elif n.rows == self.rows and n.cols == self.cols:
                 self.data = self.data + n.data
            else:
                 assert False, f"Invalid Parameters: Cannot add matrix of shape ({n.rows}, {n.cols}) to ({self.rows}, {self.cols})"
        else:
            self.data = self.data + np.float32(n)

    @staticmethod
    def transpose(m):
        if m.name is None:
            result = Matrix(m.cols, m.rows, "Result")
        else:
            result = Matrix(m.cols, m.rows, f"{m.name} (Transposed)")
        result.data = m.data.T
        return result
