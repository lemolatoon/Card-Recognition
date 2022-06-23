from typing import Tuple, List
import numpy as np
import cv2
import os
import copy
from tqdm import tqdm
import sys
from pathlib import Path
from read_excel import read_excel, BoxLabelData


def get_script_dir() -> str:
    script_dir: str = __file__
    script_dir = script_dir.split("/")
    length = len(script_dir)
    script_dir.pop(length - 1)
    script_dir = "/".join(script_dir)
    return script_dir


def main():

    # cards_path: str = f"{get_script_dir()}/../images/cards/"
    cards_path: str = f"{get_script_dir()}/../pdf2img/img/"
    # background_path: str = f"{get_script_dir()}/../images/backgrounds/"
    background_word: str = "root"
    background_path: str = f"{get_script_dir()}/../images/background/{background_word}/"

    img_pathes = np.array(list(Path(background_path).glob("*.jpg")))
    img_pathes = img_pathes[np.random.choice(img_pathes.shape[0], 1500)]

    print(img_pathes.shape)
    print(background_path)
    dir_name = background_word

    yolo = True
    # dir_name = "test"
    dataset_path: str = f"{get_script_dir()}/../images/datasets/{dir_name}/"
    dataset_yolo_path: str = f"{get_script_dir()}/../images/datasets/yolo/{dir_name}/"
    os.makedirs(f"{dataset_yolo_path}images/", exist_ok=True)
    os.makedirs(f"{dataset_yolo_path}labels/", exist_ok=True)
    images_base: str = f"{get_script_dir()}/pdf"
    images_imgs = f"{images_base}/images"
    images_lbl = f"{images_base}/labels"

    try_count: int = 1
    # box_label_list = read_excel()
    for i in range(52):
        os.makedirs(f"{dataset_path}{i}", exist_ok=True)
        box_list: List[BoxLabelData] = []
        card = cv2.imread(f"{cards_path}{i}.jpg")
        with open(f"{images_lbl}/{i}.txt", mode="r") as f:
            for line in f.readlines():
                box = []
                for val in line.split(" ")[1:]:
                    box.append(float(val))
                box_list.append(BoxLabelData.new(*box, width=card.shape[1], height=card.shape[0]))
                box_list[0]
            
        print(f"[{i+1}/53]")
        for j, img_path in enumerate(tqdm(img_pathes)):  # num background
            # print(f"(i, j): {(i, j)}")
            background = cv2.imread(str(img_path))
            for k in range(try_count):  # try affine count
                # TODO: change assert(box_label_list[i], List[Union[float, int]])
                converted, converted_labels = random_affine(
                    card, i, copy.deepcopy(background), box_list, resize=True)
                if converted is None:
                    k = k + 1
                    continue
                if yolo:
                    image_save(converted, dataset_yolo_path, i, j, k, try_count, yolo)
                    label_save(converted_labels, dataset_yolo_path, i, j, k, try_count)


def image_save(img, dataset_path: str, card_idx: int, back_idx: int, try_idx: int, n_try_count: int, yolo: bool = False):
    cv2.imwrite(im_write_path(dataset_path, card_idx, back_idx, try_idx, n_try_count, yolo), img)


def label_save(labels: BoxLabelData, dataset_path: str, card_idx: int, back_idx: int, try_idx: int, n_try_count: int):
    path = f"{dataset_path}/labels/{52 * ((back_idx) * n_try_count + try_idx) + card_idx}.txt"
    with open(path, mode="w") as f:
        for label in labels:
            for idx, param in enumerate(label.yolo_labels()):
                if idx != 0:
                    f.write(f"{param:.6f} ")
                else:
                    f.write(f"{param} ")
            f.write("\n")


def im_write_path(dataset_path: str, card_idx: int, back_idx: int, try_idx: int, n_try_count: int, yolo: bool = False) -> str:
    if yolo:
        return f"{dataset_path}/images/{52 * ((back_idx) * n_try_count + try_idx) + card_idx}.jpg"
    else:
        return f"{dataset_path}{card_idx}/{(back_idx) * n_try_count + try_idx}.jpg"


def random_affine(card_img: np.ndarray, card_label: int, background_img: np.ndarray, box_label: List[BoxLabelData], resize: bool = False, resize_length: int = 512) -> Tuple[np.ndarray, List[BoxLabelData]]:
    if background_img is None:
        return None, None
    background_img = cv2.resize(background_img, (resize_length, resize_length))

    cv2.imwrite("fig.png", card_img)
    # sys.exit(0)

    # 明度をランダムに変更
    card_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
    # v_mag = 0.7 + np.random.random()
    v_mag = 1
    card_img[:, :, (2)] = card_img[:, :, (2)] * v_mag
    card_img = cv2.cvtColor(card_img, cv2.COLOR_HSV2BGR)

    # scale 1
    s = background_img.shape[0] / (5.0 + 0.2 * (np.random.random() - 0.5)) / card_img.shape[0]
    resize_matrix = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])

    # 歪ませる
    height = background_img.shape[0]
    width = background_img.shape[1]

    z_theta = np.pi * 2 * np.random.random()
    rotate_z = np.array([[np.cos(z_theta), np.sin(z_theta), 0],
                         [-np.sin(z_theta), np.cos(z_theta), 0], [0, 0, 1.0]])
    y_theta = 0.005 * np.random.random()
    rotate_y = np.array([[np.cos(y_theta), 0, -np.sin(y_theta)],
                         [0, 1, 0], [np.sin(y_theta), 0, np.cos(y_theta)]])
    x_theta = 0.005 * np.random.random()
    rotate_x = np.array([[1, 0, 0],
                         [0, np.cos(x_theta), np.sin(x_theta)], [0, -np.sin(x_theta), np.cos(x_theta)]])
    move_x = width / 2 + width / 10 * (np.random.random() - 0.5)
    move_y = height / 2 + height / 10 * (np.random.random() - 0.5)
    move = np.array([[1, 0, move_x], [0, 1, move_y], [0, 0, 1]])

    # scale
    s = min(min(width / card_img.shape[0], height /
                card_img.shape[1]) * np.random.random() * 0.8, 1)
    scale = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])
    matrix = scale

    matrix = np.dot(np.dot(
        rotate_z, np.dot(rotate_y, rotate_x)), matrix)
    determinant = np.linalg.det(matrix)
    if determinant < 0.5:
        s = max(1 / determinant * (np.random.random() + 0.5) / 3, 1 / determinant)
        scale = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])
        matrix = np.dot(scale, matrix)
    matrix = np.dot(move, matrix)
    matrix = np.dot(matrix, resize_matrix)

    symbol_positions = [box.box_positions() for box in box_label]
    converted_symbols = cv2.perspectiveTransform(np.array(symbol_positions, dtype=np.float32), matrix)
    converted_symbols_int = np.int64(converted_symbols)
    # print(f"{symbol_positions} -> \n{converted_symbols_int}")

    dsize = (width, height)

    dst = cv2.warpPerspective(card_img, matrix, borderMode=cv2.BORDER_TRANSPARENT, dsize=dsize, dst=background_img)
    # dst = cv2.warpPerspective(
    #     card_img, matrix, flags=cv2.INTER_CUBIC, dsize=(size, size))
    # print(dst.shape)
    for converted_symbol_int in converted_symbols_int:
        for pos in converted_symbol_int:
            assert(type(pos[0]) is np.int64)
            if pos[0] < 0 or pos[0] > dst.shape[0] or pos[1] < 0 or pos[1] > dst.shape[1]:
                # return continue
                # 記号が画像に収まっていないときはNoneを返す
                return None, None
            # print("plot")
            # dst = cv2.circle(dst, (pos[0], pos[1]), radius=2, color=(0, 0, 255), thickness=3)
    labels = []
    for converted_symbol in converted_symbols:
        min_x = np.min(converted_symbol.T[0])
        max_x = np.max(converted_symbol.T[0])
        min_y = np.min(converted_symbol.T[1])
        max_y = np.max(converted_symbol.T[1])
        # dst = cv2.circle(dst, (int(min_x) - 1, int(min_y) - 1), radius=2, color=(0, 0, 255), thickness=2)
        # dst = cv2.circle(dst, (int(min_x) - 1, int(max_y) + 1), radius=2, color=(0, 0, 255), thickness=2)
        # dst = cv2.circle(dst, (int(max_x) + 1, int(min_y) - 1), radius=2, color=(0, 0, 255), thickness=2)
        # dst = cv2.circle(dst, (int(max_x) + 1, int(max_y) + 1), radius=2, color=(0, 0, 255), thickness=2)
        converted_label = BoxLabelData(min_x, min_y, max_x, max_y)
        converted_label.set_img_param(dst.shape[0], dst.shape[1], card_label)
        labels.append(converted_label)
    return dst, labels


if __name__ == "__main__":
    main()
