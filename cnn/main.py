from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
import sys


def get_script_dir() -> str:
    script_dir: str = __file__
    script_dir = script_dir.split("/")
    length = len(script_dir)
    script_dir.pop(length - 1)
    script_dir = "/".join(script_dir)
    return script_dir


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    train_dataloader, test_dataloader = get_dataloader()

    # Get a batch of training data
    images, labels = next(iter(train_dataloader))

    # Make a grid from batch
    out = torchvision.utils.make_grid(images[:3])

    imshow(out, title=[get_class_name(label) for label in labels[:3]])


def get_class_name(label: torch.Tensor) -> str:
    sign: int = label.item() // 13
    number: int = 13 - label.item() % 13
    class_name: str
    if sign == 0:
        class_name = "Spade"
    elif sign == 1:
        class_name = "Heart"
    elif sign == 2:
        class_name = "Diamond"
    elif sign == 3:
        class_name = "Club"
    else:
        print(f"label({label}) is inappropriate.")
        sys.exit(1)

    if number in range(2, 11):
        class_name += str(number)
    elif number == 1:
        class_name += "A"
    elif number == 11:
        class_name += "J"
    elif number == 12:
        class_name += "Q"
    elif number == 13:
        class_name += "K"
    else:
        print(f"Unreachable code")
        sys.exit(1)

    return class_name


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.savefig("fig.png")


def get_dataloader() -> Tuple[DataLoader, DataLoader]:
    train_dataset, test_dataset = get_dataset()

    print("Making Train DataLoader...")
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True)

    print("Making Test DataLoader...")
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=True)

    return (train_dataloader, test_dataloader)


def get_dataset() -> Tuple[Dataset, Dataset]:

    path: str = f"{get_script_dir()}/../images/datasets/"

    images = []
    labels = []
    n_image = 0
    print(path)
    print("Loading Images...")
    for i in tqdm(range(3)):
        same_label_image_paths = np.array(
            list(Path(f"{path}{i}/").glob("*.jpg")))
        length = len(same_label_image_paths)
        # メモリ足りないから半分にする
        same_label_image_paths = same_label_image_paths[np.random.choice(
            length, int(length / 3))]
        same_label_images = [np.array(Image.open(img_path))
                             for img_path in same_label_image_paths]
        print(f"image: {type(same_label_images[0])}")
        print(f"image: {same_label_images[0].shape}")
        n_image += len(same_label_images)
        images.append(same_label_images)
        labels.append([i for _ in range(len(same_label_images))])

    print(n_image)
    images = np.array(images, dtype=object)
    images = images.reshape(
        images.shape[0] * images.shape[1], images.shape[2], images.shape[3], images.shape[4])
    print(images.shape)
    print(images[0].shape)
    Image.fromarray(np.uint8(images[0]))
    images = [Image.fromarray(np.uint8(arr)) for arr in images]
    labels = np.array(labels, dtype=np.int64)
    labels = labels.reshape((-1))
    print(labels.shape)
    print(type(images[0]))
    print(type(labels[0]))
    # data = np.stack([images, labels], -1)
    data = np.array([(image, label) for image, label in zip(images, labels)])

    # randomize
    print("Shuffling data...")
    print(f"image: {type(data[0][0])}, label: {type(data[0][1])}")
    data = data[np.random.choice(data.shape[0], data.shape[0])]
    print(data.shape)

    # split
    print("Splitting train test data...")
    n_data = data.shape[0]
    train_data = data[:int(n_data * 0.8)]
    test_data = data[int(n_data * 0.8):]

    print(f"train_data.shape: {train_data.shape}")
    print(f"test_data.shape: {test_data.shape}")

    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.225])
    ])

    print("Making Dataset...")
    train_dataset = TrumpDataset(train_data, train_transform)
    test_dataset = TrumpDataset(test_data, test_transform)

    return (train_dataset, test_dataset)


class TrumpDataset(Dataset):
    def __init__(self: TrumpDataset, data: torch.Tensor, transforms) -> None:
        super().__init__()
        self.transforms = transforms
        self.data = data

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.data[index]
        image = self.transforms(data[0])
        label = torch.tensor(data[1], dtype=torch.int32)

        return (image, label)

    def __len__(self) -> int:
        return len(self.data)


if __name__ == "__main__":
    main()
