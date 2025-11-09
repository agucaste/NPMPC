import numpy as np

from typing import Tuple, Any, List
from numpy.typing import NDArray

class Cell:
    """
    
    """
    def __init__(self, center: NDArray, radius: float, split_factor: int = 3) -> None:
        self.c = center
        self.r = radius
        self.dim = center.shape[0]
        # Check that splitting number is odd.
        assert isinstance(split_factor, int) and split_factor % 2 == 1
        self.split_factor = split_factor

        self.gap: float = self.r  # the 'certification gap'. if it's > 0, the cell is un-certified
        self.is_certified: bool = False
    def __lt__(self, other):
        """
        Implements < comparison; useful for sorting lists of Cells
        """
        if not isinstance(other, Cell):
            return NotImplemented
        else:
            return self.gap < other.gap

    def certify(self, r: float) -> bool:
        """
        Checks whether the cell is 'enclosed' by a ball centered at c, with radius r.
        """
        self.gap = self.r - r
        # print(f"Gap is {self.gap}")
        self.is_certified = self.gap <= 0
        return self.is_certified
    
    
    def split(self) -> List[Any]:
        """
        Splits a cell into (split_factor^dim) sub-cells
        """
        r = self.r
        c = self.c
        s = self.split_factor
        
        new_r = self.r / self.split_factor

        z = [np.linspace(c[i] - r + r/s, c[i] + r - r/s, s) for i in range(self.dim)]
        grid = np.meshgrid(*z)
        new_centers = np.c_[tuple(g.ravel() for g in grid)]

        return [Cell(new_c, new_r, self.split_factor) for new_c in new_centers]
    
# class CellColection(Cell):
#     """
#     A collection of cells.
#     """
#     def __init__(self, cells: Cell | List[Cell]):        
#         self.cells = cells if isinstance(cells, list[Cell]) else self.cells = [cells]
        
            

    

    
if __name__ == "__main__":
    cell = Cell(center=np.array([0.0, 0.0]), radius=3.0, split_factor=3)
    centers = cell.split()
    print("New centers:")
    print(centers)
    
    
