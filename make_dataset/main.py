import numpy as np
import cv2
import os
import copy


def get_script_dir() -> str:
    script_dir: str = __file__
    script_dir = script_dir.split("/")
    length = len(script_dir)
    script_dir.pop(length - 1)
    script_dir = "/".join(script_dir)
    return script_dir


def main():
    # test
    card = cv2.imread(f"{get_script_dir()}/../images/cards/15.jpg")
    background = cv2.imread(
        f"{get_script_dir()}/../images/backgrounds/000002.jpg")
    affined = random_affine(card, background)
    cv2.imwrite("fig.png", background)
    cv2.imwrite("fig.png", affined)
    cv2.imwrite("fig.png", background)

    cards_path: str = f"{get_script_dir()}/../images/cards/"
    background_path: str = f"{get_script_dir()}/../images/backgrounds/"
    dataset_path: str = f"{get_script_dir()}/../images/datasets/"
    for i in range(53):
        card = cv2.imread(f"{cards_path}{4 + i}.jpg")
        for j in range(1, 400):  # num background
            print(f"(i, j): {(i, j)}")
            background = cv2.imread(f"{background_path}{j:06}.jpg")
            for k in range(10):  # try affine count
                converted = random_affine(copy.deepcopy(
                    card), copy.deepcopy(background))
                os.makedirs(f"{dataset_path}{i}", exist_ok=True)
                print(f"{dataset_path}{i}/{j * 100 + k}.jpg")
                cv2.imwrite(f"{dataset_path}{i}/{j * 100 + k}.jpg", converted)


def random_affine(card_img: np.ndarray, background_img: np.ndarray) -> np.ndarray:
    matrix = np.array(
        [1 + np.random.random() * 0.005 if i in (0, 4, 8) else np.random.random() * 0.0005 for i in range(9)]).reshape((3, 3))

    src_pts = np.array(
        [[0, 0], [0, card_img.shape[1]], [card_img.shape[0], 0], [card_img.shape[0], card_img.shape[1]]], dtype=np.float32)

    def gen_x(): return card_img.shape[0] + \
        card_img.shape[0] * 0.1 * np.random.random() * 20

    def gen_y(): return card_img.shape[1] + \
        card_img.shape[1] * 0.1 * np.random.random() * 20
    dst_pts = np.array(
        [[0, 0], [np.random.random() * 20, gen_y()], [gen_x(), np.random.random() * 20], [gen_x(), gen_y()]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    height = background_img.shape[0]
    width = background_img.shape[1]
    matrix = matrix * min(height, width) / \
        max(card_img.shape[0], card_img.shape[1]) * 0.70

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
    dx = width / 2 + (np.random.random() - 0.5) * width / 5
    dy = height / 2 + (np.random.random() - 0.5) * height / 5
    matrix[0][2] = dx
    matrix[1][2] = dy

    # matrix = matrix / \
    # np.linalg.det(matrix) * (0.5 + np.random.random()) * \
    # 0.1 * min(height, width) / \
    # np.sqrt(card_img.shape[0] ** 2 + card_img.shape[1] ** 2)
    matrix = np.dot(move, np.dot(scale, np.dot(
        rotate_z, np.dot(rotate_y, rotate_x))))
    dsize = (width, height)

    dst = cv2.warpPerspective(card_img, matrix, borderMode=cv2.BORDER_TRANSPARENT,
                              dsize=dsize, dst=background_img)
    # dst = cv2.warpPerspective(
    #     card_img, matrix, flags=cv2.INTER_CUBIC, dsize=(size, size))
    return dst


if __name__ == "__main__":
    main()
