import openpyxl
from typing import List, Optional, Tuple, Union
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
    def __init__(self, min_x: Union[int, float], min_y: Union[int, float], max_x: Union[int, float], max_y: Union[int, float]):
        self.labels: List[Union[int, float]] = [min_x, min_y, max_x, max_y]
        self.width = None
        self.height = None
        self.class_label = None

    def box_positions(self) -> List[np.ndarray]:
        positions = []
        positions.append(np.array([self.labels[0], self.labels[1]], dtype=np.float32))  # left up
        positions.append(np.array([self.labels[2], self.labels[1]], dtype=np.float32))  # right up
        positions.append(np.array([self.labels[0], self.labels[3]], dtype=np.float32))  # left down
        positions.append(np.array([self.labels[2], self.labels[3]], dtype=np.float32))  # right down

        return positions

    def set_img_param(self, width: int, height: int, class_label: int):
        self.width = width
        self.height = height
        self.class_label = class_label

    def yolo_labels(self) -> Optional[List[Union[int, float]]]:
        if self.width is None or self.height is None or self.class_label is None:
            return None
        label: List[Union[int, float]] = []
        assert(type(self.class_label) is int)
        label.append(self.class_label)  # class
        label.append((self.labels[0] + self.labels[2]) / 2.0)  # x_center
        label.append((self.labels[1] + self.labels[3]) / 2.0)  # y_center
        # Note: have to normalize
        label.append((self.labels[2] - self.labels[0]) * 1.0 / self.width)  # width
        label.append((self.labels[3] - self.labels[1]) * 1.0 / self.height)  # height

        return label


if __name__ == "__main__":
    read_excel()
