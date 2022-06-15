import numpy as np
import cv2
import os
import copy
from tqdm import tqdm
import sys
from pathlib import Path


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
    # dataset_path: str = f"{get_script_dir()}/../images/datasets/"
    # dataset_path: str = f"{get_script_dir()}/../images/datasets_desk/"
    dataset_path: str = f"{get_script_dir()}/../images/datasets/{dir_name}/"
    try_count: int = 1
    for i in range(52):
        os.makedirs(f"{dataset_path}{i}", exist_ok=True)
        card = cv2.imread(f"{cards_path}{i}.jpg")
        print(f"[{i+1}/53]")
        img_path = "{get_script_dir()}/../images/background/{background_word}/001988.png"
        print(img_path)
        # print(f"(i, j): {(i, j)}")
        background = cv2.imread(str(img_path))
        converted = random_affine(
            card, copy.deepcopy(background),i, resize=True)
        if converted is None:
            continue
        # print(f"{dataset_path}{i}/{j * 100 + k}.jpg")


def random_affine(card_img: np.ndarray, background_img: np.ndarray, ite:int,resize: bool = False, resize_length: int = 255) -> np.ndarray:
    simbol_pos = [(),(),(),()]
    if background_img is None:
        return None
    background_img = cv2.resize(background_img, (500, 500))

    # reshape
    rate = background_img.shape[0] / (3.5 + 0.2 * (np.random.random() - 0.5)) / card_img.shape[0]

    #トランプの画像のサイズ？
    dsize = (int(card_img.shape[1] * rate), int(card_img.shape[0] * rate))
    card_img = cv2.resize(card_img, dsize)
    cv2.imwrite(f"fig{ite}.png", card_img)
    # sys.exit(0)

if __name__ == "__main__":
    main()