from pathlib import Path
import numpy as np
import os
import shutil
from tqdm import tqdm


def get_script_dir() -> str:
    script_dir: str = __file__
    script_dir = script_dir.split("/")
    length = len(script_dir)
    script_dir.pop(length - 1)
    script_dir = "/".join(script_dir)
    return script_dir


def main():
    yolo_base_path: str = f"{get_script_dir()}/../images/datasets/yolo"
    dir_name: str = "real_root"
    yolo_base_root_path: str = f"{yolo_base_path}/{dir_name}"
    image_base_root: str = f"{yolo_base_root_path}/images"
    labels_base_root: str = f"{yolo_base_root_path}/labels"
    images_files = np.array(list(Path(image_base_root).glob("*.jpg")))
    labels_files = np.array(list(Path(labels_base_root).glob("*.txt")))

    print(yolo_base_path)
    print(images_files.shape)
    print(labels_files.shape)
    n_data = np.min([images_files.shape[0], labels_files.shape[0]])
    idx_list = np.random.choice(n_data, n_data)
    rate: float = 0.05
    val_idx = idx_list[:int(n_data * rate)]
    train_idx = idx_list[int(n_data * rate):]

    val_images = images_files[val_idx]
    val_labels = [Path(f"{labels_base_root}/{img_path.stem}.txt") for img_path in val_images]

    train_images = images_files[train_idx]
    train_labels = [Path(f"{labels_base_root}/{img_path.stem}.txt") for img_path in train_images]

    yolo_val_base: str = f"{yolo_base_path}/val"
    yolo_train_base: str = f"{yolo_base_path}/train"
    os.makedirs(f"{yolo_val_base}/images", exist_ok=True)
    os.makedirs(f"{yolo_val_base}/labels", exist_ok=True)
    os.makedirs(f"{yolo_train_base}/images", exist_ok=True)
    os.makedirs(f"{yolo_train_base}/labels", exist_ok=True)

    print("Creating validation data...")
    for v_img, v_lbl in tqdm(zip(val_images, val_labels)):
        try:
            shutil.copy(str(v_img), f"{yolo_val_base}/images/{v_img.stem}.jpg")
        except:
            print(f"\n{yolo_val_base}/images/{v_img.stem}.jpg is missing")
        try:
            shutil.copy(str(v_lbl), f"{yolo_val_base}/labels/{v_lbl.stem}.txt")
        except:
            print(f"\n{yolo_val_base}/labels/{v_lbl.stem}.txt is missing")

    print("Creating train data...")
    for t_img, t_lbl in tqdm(zip(train_images, train_labels)):
        try:
            shutil.copy(str(t_img), f"{yolo_train_base}/images/{t_img.stem}.jpg")
        except:
            print(f"\n{yolo_train_base}/images/{t_img.stem}.jpg is missing")
        try:
            shutil.copy(str(t_lbl), f"{yolo_train_base}/labels/{t_lbl.stem}.txt")
        except:
            print(f"\n{yolo_train_base}/labels/{t_lbl.stem}.txt is missing")


if __name__ == "__main__":
    main()
