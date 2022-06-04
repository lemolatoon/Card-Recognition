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
            for k in range(100):  # try affine count
                converted = random_affine(copy.deepcopy(
                    card), copy.deepcopy(background))
                os.makedirs(f"{dataset_path}{i}", exist_ok=True)
                print(f"{dataset_path}{i}/{j * k + k}.jpg")
                cv2.imwrite(f"{dataset_path}{i}/{j * k + k}.jpg", converted)


def random_affine(card_img: np.ndarray, background_img: np.ndarray) -> np.ndarray:
    matrix = np.array(
        [1 + np.random.random() * 0.005 if i in (0, 4, 8) else np.random.random() * 0.0005 for i in range(9)]).reshape((3, 3))

    src_pts = np.array(
        [[0, 0], [0, card_img.shape[1]], [card_img.shape[0], 0]], dtype=np.float32)

    def gen_x(): return card_img.shape[0] + \
        card_img.shape[0] * 0.1 * np.random.random() * 30

    def gen_y(): return card_img.shape[1] + \
        card_img.shape[1] * 0.1 * np.random.random() * 30
    dst_pts = np.array(
        [[0, 0], [np.random.random() * 30, gen_y()], [gen_x(), np.random.random() * 30]], dtype=np.float32)
    matrix = cv2.getAffineTransform(src_pts, dst_pts)
    height = background_img.shape[0]
    width = background_img.shape[1]
    matrix = matrix * 1.2 * min(height, width) * \
        max(card_img.shape[0], card_img.shape[1]) / 100

    rotate_mat = cv2.getRotationMatrix2D(
        (0.0, 0.0), np.random.random() * 360, 1.0)
    matrix += rotate_mat
    # matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    size = max(card_img.shape[0], card_img.shape[1]) * 3 + 50
    dx = (np.random.random() - 0.5) * width / 3 + width / 3
    dy = (np.random.random() - 0.5) * height / 3 + height / 3
    matrix[0][2] = dx
    matrix[1][2] = dy
    dsize = (width, height)

    dst = cv2.warpAffine(card_img, matrix, borderMode=cv2.BORDER_TRANSPARENT,
                         dsize=dsize, dst=background_img)
    # dst = cv2.warpPerspective(
    #     card_img, matrix, flags=cv2.INTER_CUBIC, dsize=(size, size))
    return dst


if __name__ == "__main__":
    main()
