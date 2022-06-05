import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyheif
from PIL import Image
import sys


def get_script_dir() -> str:
    script_dir: str = __file__
    script_dir = script_dir.split("/")
    length = len(script_dir)
    script_dir.pop(length - 1)
    script_dir = "/".join(script_dir)
    return script_dir


def main():
    name: str = "IMG_5492.HEIC"
    path = f"{get_script_dir()}/../images/{name}"
    # HEICファイル形式の場合はこれを呼ぶ
    # img = heic2png(path)
    name: str = "cards.png"
    # bgr format
    img: np.ndarray = cv2.imread(f"{get_script_dir()}/../images/{name}")
    print(type(img))
    print(img.shape)
    # rgb format
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    plt.savefig("fig.png")
    # gray scale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rbg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_gray)
    plt.savefig("fig.png")

    ret, bin_img = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    plt.imshow(bin_img)
    plt.savefig("fig.png")

    # 輪郭検出
    contours = edge_detection(bin_img)

    # 直線検出
    for idx, cnt in enumerate(contours):
        approx: np.ndarray = cv2.approxPolyDP(
            cnt, 0.01 * cv2.arcLength(cnt, True), True)
        # print(approx)
        # print(approx.shape)
        # print(type(approx))
        if approx.size != 8:
            # 四角形以外は排除
            continue
        # cv2.drawContours(img, [approx], -1, color=(0, 0, 255), thickness=2)

        # print(approx.shape)
        approx = approx[:, 0, :]
        # print(approx.shape)
        y_1 = min(approx[0][0], approx[1][0])  # left up
        y_2 = max(approx[2][0], approx[3][0])  # left down
        x_1 = min(approx[0][1], approx[1][1])  # right up
        x_2 = max(approx[2][1], approx[3][1])  # right down
        # print(f"x_1: {x_1}, x_2: {x_2}, y_1: {y_1}, y_1, {y_2}")
        # print(img.shape)
        # 輪郭の四角形ごとにcard変数に保存し、画像としてwrite
        card: np.ndarray = img[x_1:x_2, y_1:y_2, :]
        # cv2.imwrite(f"{get_script_dir()}/../images/cards/{idx}.jpg", card)
        # 4 + 13 * 0 ~ 4 + 13 * 1 - 1 -> スペード K ~ A
        # 4 + 13 * 1 ~ 4 + 13 * 2 - 1 -> ハート K ~ A
        # 4 + 13 * 2 ~ 4 + 13 * 3 - 1 -> ダイヤ K ~ A
        # 4 + 13 * 3 ~ 4 + 13 * 4 - 1 -> クラブ K ~ A

    # 輪郭描画
    cv2.drawContours(img, contours, -1, color=(0, 0, 255), thickness=2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_rbg)
    plt.savefig("fig.png")


def edge_detection(img: np.ndarray):
    img.reshape((img.shape[0], img.shape[1], 1))
    print(img.shape)
    contours, _ = cv2.findContours(
        # img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(len(contours))
    # contours = np.array(contours, dtype=object)  # tuple to ndarray
    print(type(contours))
    print(type(contours[0]))
    print(type(contours[0][0]))
    print(type(contours[0][0][0]))
    print(type(contours[0][0][0]))
    print(type(contours[0][0][0][0]))
    return contours


def heic2png(img_path: str) -> np.ndarray:
    heif_file = pyheif.read(img_path)
    data = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    print(type(data))
    return np.array(data)


if __name__ == "__main__":
    main()
