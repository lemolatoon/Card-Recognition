from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image
import pyheif
from pathlib import Path
import pathlib
from tqdm import tqdm
import os
import copy


def main():
    test()


def test():
    path = f"{get_script_dir()}/../images/competition_sample"
    images: List[np.ndarray]
    if Path.exists(Path(f"{path}/jpg")):
        path = f"{path}/jpg"
        same_label_image_paths = np.array(
            list(Path(f"{path}").glob("*")))
        images = get_images(same_label_image_paths)
    else:
        same_label_image_paths = np.array(
            list(Path(f"{path}").glob("*")))
        images = get_heic_images(same_label_image_paths)

    for idx, image in enumerate(images):
        x, y, w, h = image_to_square(image)
        ROI = image[y: y + h, x: x + w]
        cv2.imwrite(f"ROI_{idx}.jpg", ROI)
        return


def get_heic_images(pathes: List[pathlib.PosixPath]) -> List[np.ndarray]:
    images = []
    for idx, heic_file in enumerate(pathes):
        if Path.is_dir(heic_file):
            continue
        image = np.array(heic2png(heic_file))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        images.append(image)
    return images


def get_images(pathes: List[pathlib.PosixPath]) -> List[np.ndarray]:
    images = []
    for idx, heic_file in enumerate(pathes):
        if Path.is_dir(heic_file):
            continue
        image = cv2.imread(str(heic_file))
        images.append(image)
    return images


def image_to_square(image: np.ndarray) -> Tuple[int, int, int, int]:
    # detected square must consist of 4 params, returns the 4 params
    # x, y, w, h
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    currently_biggest_area = 0
    min_area = image.shape[0] * image.shape[1] / 25
    max_area = image.shape[0] * image.shape[1] / 8
    params = None

    threshold_max = 255
    os.makedirs(f"{get_script_dir()}/tmp", exist_ok=True)
    for idx, threshold in enumerate(tqdm(range(1, threshold_max))):
        thresh = cv2.threshold(sharpen, threshold, 255, cv2.THRESH_BINARY_INV)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            # draw line
            arclen = cv2.arcLength(c, True)
            approx: np.ndarray = cv2.approxPolyDP(c, arclen * 0.02, True)
            # print(approx)

            # calc area
            area = abs(cv2.contourArea(approx))
            if area < currently_biggest_area or approx.shape[0] != 4:
                continue
            currently_biggest_area = area
            if area > min_area and area < max_area:
                params = [param for param in cv2.boundingRect(c)]
            image_with_line = copy.deepcopy(image)

            image_with_line = cv2.polylines(image_with_line, c, True, (0, 0, 255),
                                            thickness=5, lineType=cv2.LINE_8)
            for pt in approx:
                pos = pt[0]
                cv2.drawMarker(image_with_line, (pos[0], pos[1]), (0, 255, 0), thickness=4)
        cv2.imwrite(f"{get_script_dir()}/tmp/{idx}.jpg", image_with_line)

    if params is not None:
        print("OK")
        return params[0], params[1], params[2], params[3]
    print("Not OK")
    return 1, 1, 1, 1


def heic2png(img_path: str) -> Image:
    heif_file = pyheif.read(img_path)
    data = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    return data


def get_script_dir() -> str:
    script_dir: str = __file__
    script_dir = script_dir.split("/")
    length = len(script_dir)
    script_dir.pop(length - 1)
    script_dir = "/".join(script_dir)
    return script_dir


if __name__ == "__main__":
    main()
