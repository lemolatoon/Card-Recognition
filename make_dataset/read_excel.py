import openpyxl
from typing import List, Optional, Tuple
import numpy as np


def get_script_dir() -> str:
    script_dir: str = __file__
    script_dir = script_dir.split("/")
    length = len(script_dir)
    script_dir.pop(length - 1)
    script_dir = "/".join(script_dir)
    return script_dir


def read_excel(path: Optional[str] = None):
    book_path: str
    if path is None:
        book_name: str = "card_book"
        book_path: str = f"{get_script_dir()}/{book_name}.xlsx"
    else:
        book_path = path

    card_book = openpyxl.load_workbook(book_path)
    card_book = card_book["Sheet1"]
    label_positions: List[BoxLabelData] = []
    for col in card_book.iter_rows():
        labels: List[int] = []
        for cell in col:
            print(f"{cell.value}", end=" ")
            assert(type(cell.value) is int)
            labels.append(cell.value)
        print()
        label_data = BoxLabelData(*labels)
        label_positions.append(label_data)

    return label_positions


class BoxLabelData:
    def __init__(self, min_x: int, min_y: int, max_x: int, max_y: int):
        self.labels: List[int] = [min_x, min_y, max_x, max_y]

    def box_positions(self) -> List[np.ndarray]:
        positions = []
        positions.append(np.array([self.labels[0], self.labels[1]], dtype=np.float32))  # left up
        positions.append(np.array([self.labels[2], self.labels[1]], dtype=np.float32))  # right up
        positions.append(np.array([self.labels[0], self.labels[3]], dtype=np.float32))  # left down
        positions.append(np.array([self.labels[2], self.labels[3]], dtype=np.float32))  # right down

        return positions


if __name__ == "__main__":
    read_excel()
