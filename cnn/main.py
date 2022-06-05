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
import copy
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models
import pandas as pd
import pyheif


def get_script_dir() -> str:
    script_dir: str = __file__
    script_dir = script_dir.split("/")
    length = len(script_dir)
    script_dir.pop(length - 1)
    script_dir = "/".join(script_dir)
    return script_dir


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(train_loader: DataLoader, test_loader: DataLoader, model: nn.Module, criterion, optimizer: optim.Optimizer, scheduler, num_epochs: int = 100):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # iter, train_loss, train_acc, val_loss, val_acc
    batch_size: int = iter(train_loader).next()[0].shape[0]
    history = np.zeros(5)

    for epoch in range(num_epochs):
        train_acc, train_loss = 0.0, 0.0
        val_acc, val_loss = 0.0, 0.0
        num_train, num_test = 0, 0

        model.train()
        inputs: torch.Tensor
        labels: torch.Tensor
        for inputs, labels in tqdm(train_loader):
            num_train += len(labels)

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs: torch.Tensor = model(inputs).to(device)
            _, predicted = torch.max(outputs, 1)
            loss: torch.Tensor = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (predicted == labels).sum().item()

            # scheduler.step()

        inputs_test: torch.Tensor
        labels_test: torch.Tensor
        for inputs_test, labels_test in test_loader:
            model.eval()
            num_test += len(labels_test)
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)

            with torch.set_grad_enabled(False):
                outputs_test = model(inputs_test)
                _, predicted_test = torch.max(outputs_test, 1)
                loss_test: torch.Tensor = criterion(outputs_test, labels_test)

            val_loss += loss_test.item()
            val_acc += (predicted_test == labels_test).sum().item()

        train_acc = train_acc / num_train
        val_acc = val_acc / num_test
        train_loss = train_loss * batch_size / num_train
        val_loss = val_loss * batch_size / num_test

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}")
        items = np.array([epoch + 1, train_loss, train_acc, val_loss, val_acc])
        history = np.vstack((history, items))

    print(f"Best val Acc: {best_acc:.5f}")
    model.load_state_dict(best_model_wts)
    return model, history


def main():
    train_dataloader, test_dataloader = get_dataloader()

    # Get a batch of training data
    images, labels = next(iter(train_dataloader))

    # Make a grid from batch
    out = torchvision.utils.make_grid(images[:3])

    imshow(out, title=[get_class_name(label) for label in labels[:3]])

    model_ft = get_model()
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)

    print(device)
    model_ft, history = train_model(train_dataloader, test_dataloader,
                                    model_ft, criterion, optimizer_ft, exp_lr_scheduler)
    torch.save(model_ft.state_dict(), "param.pt")
    pd.to_pickle(history, "history.pkl")


def get_model() -> nn.Module:
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # 52 classify
    model_ft.fc = ModelHead(num_ftrs, 256, 52)

    model_ft.to(device)
    return model_ft


class ModelHead(nn.Module):
    def __init__(self, num_input: int, num_hidden: int, num_output: int):
        super().__init__()
        self.fc1 = nn.Linear(num_input, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def visualize_model(test_loader, model: nn.Module, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {get_class_name(preds[j])}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


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
    for i in tqdm(range(52)):
        same_label_image_paths = np.array(
            list(Path(f"{path}{i}/").glob("*.jpg")))
        length = len(same_label_image_paths)
        # メモリ足りないから半分にする
        same_label_image_paths = same_label_image_paths[np.random.choice(
            length, int(length / 15))]
        same_label_images = [np.array(Image.open(img_path))
                             for img_path in same_label_image_paths]
        # print(f"image: {type(same_label_images[0])}")
        # print(f"image: {same_label_images[0].shape}")
        n_image += len(same_label_images)
        images.append(same_label_images)
        labels.append([i for _ in range(len(same_label_images))])

    print(n_image)
    images = np.array(images)
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
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(128),
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
        label = torch.tensor(data[1], dtype=torch.int64)

        return (image, label)

    def __len__(self) -> int:
        return len(self.data)


def check_history():
    # [[index, train_loss, train_acc, val_loss, val_acc]], len(history) == num_epoch + 1, len(history[0]) == 5
    history: np.ndarray = pd.read_pickle("history.pkl")
    plt.plot(history[1:, 0], history[1:, 1], "b", label="train")
    plt.plot(history[1:, 0], history[1:, 3], "k", label="test")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss curve")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.close()

    plt.plot(history[:, 0], history[:, 2], "b", label="train")
    plt.plot(history[:, 0], history[:, 4], "k", label="test")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("accuracy")
    plt.legend()
    plt.savefig("acc_curve.png")

    print(f"initial: loss: {history[1, 3]}, acc: {history[1, 4]}")
    print(f"last: loss: {history[-1, 3]}, acc: {history[-1, 4]}")
    print(f"eval mode acc: {check_acc()}")

    model = load_model()
    _, test_loader = get_dataloader()
    visualize_model(test_loader, model)


def check_acc():
    net = load_model()
    _, test_loader = get_dataloader()
    acc: float = 0.0
    num_test: int = 0
    images: torch.Tensor
    labels: torch.Tensor
    for images, labels in test_loader:
        num_test += len(labels)
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        acc += (predicted == labels).sum().item()
    return acc / num_test


def check_my_img():
    model = load_model()
    path = f"{get_script_dir()}/../images/competition_sample"
    same_label_image_paths = np.array(
        list(Path(f"{path}").glob("*.HEIC")))
    images = []
    for heic_file in same_label_image_paths:
        # HEICファイル形式の場合はこれを呼ぶ
        images.append(heic2png(heic_file))
    num_images = len(images)

    test_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.225])
    ])

    inputs = [test_transform(img) for img in images]
    inputs = torch.stack(inputs)

    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    with torch.no_grad():
        inputs = inputs.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title(f'predicted: {get_class_name(preds[j])}')
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                model.train(mode=was_training)
                return


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
    return data


def load_model() -> nn.Module:
    model = get_model()
    model.load_state_dict(torch.load("param.pt"))
    model.eval()
    return model


if __name__ == "__main__":
    main()
    # check_history()
    # check_my_img()
