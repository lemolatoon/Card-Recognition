import numpy as np
import cv2
import os
import copy
from tqdm import tqdm
import sys


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
    background_word: str = "floor"
    background_path: str = f"{get_script_dir()}/../images/{background_word}/"
    print(background_path)
    dir_name = background_word
    # dataset_path: str = f"{get_script_dir()}/../images/datasets/"
    # dataset_path: str = f"{get_script_dir()}/../images/datasets_desk/"
    dataset_path: str = f"{get_script_dir()}/../images/datasets/{dir_name}/"
    try_count: int = 6
    for i in range(52):
        os.makedirs(f"{dataset_path}{i}", exist_ok=True)
        card = cv2.imread(f"{cards_path}{i}.jpg")
        print(f"[{i}/53]")
        for j in tqdm(range(1, 400)):  # num background
            # print(f"(i, j): {(i, j)}")
            background = cv2.imread(f"{background_path}{j:06}.jpg")
            for k in range(try_count):  # try affine count
                converted = random_affine(
                    card, copy.deepcopy(background), resize=True)
                if converted is None:
                    k = k + 1
                    continue
                # print(f"{dataset_path}{i}/{j * 100 + k}.jpg")
                cv2.imwrite(
                    f"{dataset_path}{i}/{(j - 1) * try_count + k}.jpg", converted)


def random_affine(card_img: np.ndarray, background_img: np.ndarray, resize: bool = False, resize_length: int = 255) -> np.ndarray:

    if background_img is None:
        return None
    background_img = cv2.resize(background_img, (500, 500))

    # reshape
    rate = background_img.shape[0] / \
        (3 + 0.2 * (np.random.random() - 0.5)) / card_img.shape[0]
    dsize = (int(card_img.shape[1] * rate), int(card_img.shape[0] * rate))
    card_img = cv2.resize(card_img, dsize)
    cv2.imwrite("fig.png", card_img)
    # sys.exit(0)

    # 明度をランダムに変更
    card_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
    v_mag = 0.5 + np.random.random()
    card_img[:, :, (2)] = card_img[:, :, (2)] * v_mag
    # for _ in range(3):
    #     h_deg = 0  # 色相
    #     s_mag = min(max(np.random.random() * 1.3, 1.1), 0.5)  # 彩度
    #     v_mag = max(min(np.random.random() * 0.8, 0.8), 1.1)  # 明度
    #     w_choice = np.random.choice(card_img.shape[0], card_img.shape[0] // 7)
    #     h_choice = np.random.choice(card_img.shape[1], card_img.shape[1] // 7)
    #     card_img[:, h_choice,
    #              (2)] = card_img[:, h_choice, (2)] * v_mag
    #     card_img[w_choice, :,
    #              (2)] = card_img[w_choice, :, (2)] * v_mag

    card_img = cv2.cvtColor(card_img, cv2.COLOR_HSV2BGR)

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

    s = min(min(width / card_img.shape[0], height /
                card_img.shape[1]) * np.random.random() * 0.8, 1)
    scale = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])
    # matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    # matrix = matrix / \
    # np.linalg.det(matrix) * (0.5 + np.random.random()) * \
    # 0.1 * min(height, width) / \
    # np.sqrt(card_img.shape[0] ** 2 + card_img.shape[1] ** 2)
    matrix = np.dot(scale, np.dot(
        rotate_z, np.dot(rotate_y, rotate_x)))
    determinant = np.linalg.det(matrix)
    if determinant < 0.5:
        s = max(1 / determinant * (np.random.random() + 0.5) / 3, 1 / determinant)
        scale = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])
        matrix = np.dot(scale, matrix)
    matrix = np.dot(move, matrix)

    dsize = (width, height)

    dst = cv2.warpPerspective(card_img, matrix, borderMode=cv2.BORDER_TRANSPARENT,
                              dsize=dsize, dst=background_img)
    # dst = cv2.warpPerspective(
    #     card_img, matrix, flags=cv2.INTER_CUBIC, dsize=(size, size))
    if resize:
        dst = cv2.resize(dst, (resize_length, resize_length))
    return dst


if __name__ == "__main__":
    main()
