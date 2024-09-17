from typing import Callable, Dict, Tuple, List
from scipy.sparse import csr_matrix

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

    def get_row(self, row: int) -> List[float]:
        """Get all values for a specific row."""
        return [self.data.get((row, col), 0) for col in range(self.max_col)]

    def row_sum(self, row: int) -> float:
        total = 0
        for col in range(self.max_col):
            value = self.data.get((row, col), 0)
            if value != 0:
                total += value
        return total

    def get_column(self, col: int) -> List[float]:
        return [self.data.get((row, col), 0) for row in range(self.max_row)]

    @DeprecationWarning
    def to_csr(self) -> csr_matrix:
        if not self.data:
            return csr_matrix((0, 0))
        
        rows, cols = zip(*self.data.keys())
        values = list(self.data.values())
        return csr_matrix((values, (rows, cols)), shape=(self.max_row, self.max_col))
