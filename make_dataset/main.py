import numpy as np
import cv2
import sys


def get_script_dir() -> str:
    script_dir: str = __file__
    script_dir = script_dir.split("/")
    length = len(script_dir)
    script_dir.pop(length - 1)
    script_dir = "/".join(script_dir)
    return script_dir


def main():
    card = cv2.imread(f"{get_script_dir()}/../images/cards/15.jpg")
    background = cv2.imread(
        f"{get_script_dir()}/../images/backgrounds/000002.jpg")
    affined = random_affine(card, background)
    cv2.imwrite("fig.png", background)
    cv2.imwrite("fig.png", affined)
    cv2.imwrite("fig.png", background)


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
    matrix = matrix * 0.9
    print(src_pts)
    print(matrix.dtype)
    # matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    print(matrix)
    print(matrix.shape)
    height = background_img.shape[0]
    width = background_img.shape[1]
    print(card_img.shape)
    size = max(card_img.shape[0], card_img.shape[1]) * 3 + 50
    dx = (np.random.random() - 0.5) * width + width / 2
    dy = (np.random.random() - 0.5) * height + height / 2
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
