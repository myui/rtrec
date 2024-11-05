from typing import Callable, Dict, Tuple, List, Optional

class SparseMatrix:
    """
    Row-major sparse matrix implementation.
    """

    def __init__(self) -> None:
        self.data: Dict[int, Dict[int, float]] = {}
        self.nnz: int = 0

    def __setitem__(self, key: Tuple[int, int], value: float) -> None:
        if value == 0:
            self.__delitem__(key)
            return

        i, j = key
        if value != 0.0:
            cols = self.data.get(i, None)
            if cols is None:
                cols = {}
                self.data[i] = cols
            cols[j] = value
            self.nnz += 1

    def __getitem__(self, key: Tuple[int, int]) -> float:
        i, j = key
        cols = self.data.get(i, None)
        if cols is not None:
            return cols.get(j, 0.0)
        return 0.0

    def __delitem__(self, key: Tuple[int, int]) -> Optional[int]:
        i, j = key
        cols = self.data.get(i, None)
        if cols is not None:
            ret = cols.pop(j, None)
            if ret is not None:
                self.nnz -= 1
                if not cols: # Remove the row if it's empty
                    del self.data[i]
                return ret
        return None

    def __repr__(self) -> str:
        return f"SparseMatrix(data={self.data}, nnz={self.nnz})"

    def __iter__(self):
        raise NotImplementedError("Iteration is not supported for SparseMatrix")

    def len(self) -> int:
        return self.nnz

    def get(self, key: Tuple[int, int], default: float = 0.0) -> float:
        i, j = key
        cols = self.data.get(i, None)
        if cols is not None:
            return cols.get(j, default)
        return default

    def get_row(self, row: int) -> Dict[int, float]:
        return self.data.get(row, {})

    def pop(self, key: Tuple[int, int], default: Optional[float] = None) -> Optional[float]:
        return self.__delitem__(key)

    def row_sum(self, row: int) -> float:
        total = 0
        cols = self.data.get(row, None)
        if cols is not None:
            for value in cols.values():
                total += value
        return total

class DoKMatrix:
    def __init__(self) -> None:
        self.data: Dict[Tuple[int, int], float] = {}
        self.max_row: int = 0
        self.max_col: int = 0

    def __setitem__(self, key: Tuple[int, int], value: float) -> None:
        i, j = key
        if value != 0:
            self.data[(i, j)] = value
            self.max_row = max(self.max_row, i + 1)
            self.max_col = max(self.max_col, j + 1)
        elif (i, j) in self.data:
            del self.data[(i, j)]
    
    def __getitem__(self, key: Tuple[int, int]) -> float:
        return self.data.get(key, 0)
    
    def __delitem__(self, key: Tuple[int, int]) -> None:
        self.data.pop(key, None)

    def __repr__(self) -> str:
        return f"DoKMatrix(data={self.data}, shape=({self.max_row}, {self.max_col}))"

    def __iter__(self):
        return iter(self.data.items())

    def len(self) -> int:
        return len(self.data)

    def get(self, key: Tuple[int, int], default: float = 0) -> float:
        return self.data.get(key, default)

    def row_sum(self, row: int) -> float:
        total = 0
        for col in range(self.max_col):
            value = self.data.get((row, col), 0)
            if value != 0:
                total += value
        return total

    def get_column(self, col: int) -> List[float]:
        return [self.data.get((row, col), 0) for row in range(self.max_row)]
