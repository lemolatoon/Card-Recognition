from typing import List, Tuple
import math
import os
import dis
import sys

def get_script_dir() -> str:
    script_dir: str = __file__
    if script_dir == "main.py":
        return "."
    script_dir = script_dir.split("/")
    length = len(script_dir)
    script_dir.pop(length - 1)
    script_dir = "/".join(script_dir)
    return script_dir

def main():
    N_k: List[int] = [0] * 13
    
    print(get_script_dir())
    label_path = f"{get_script_dir()}/../competi/images/label.txt"
    out_labels_path = f"{get_script_dir()}/../competi/out/exp/labels/"
    stem_label: List[Tuple[int, int]] = []
    with open(f"{label_path}", mode="r") as f:
        lines = f.readlines()
    print(lines)
    for line in lines:
        line = line.strip()
        line = line.split(" ")
        stem = line[0]

        n_card = card_num(line[1])
        N_k[n_card] += 1

        label = class2label(line[1])
        stem_label.append((stem, label))
    print(stem_label)

    n_correct = 0
    n_case = 0
    for stem, label in stem_label:
        path = out_labels_path + stem + ".txt"
        with open(path, mode="r") as f:
            lines = f.readlines()
        conf = 0
        for line in lines:
            line = line.strip()
            line = line.split(" ")
            # pred_label2 = line.index(max(conf, float(line[5])))[0]
            conf = 0
            if float(line[5]) > conf:
                conf = line[5]
                pred_label= line[0]
        
        print(f"pred_label: {pred_label}, label: {label}")
        if int(pred_label) == label:
            n_correct += 1
        n_case += 1

    print(f"n_correct: {n_correct}, n_case: {n_case}")
    acc = n_correct / n_case
    print(f"acc: {acc}")
    print(f"N_k: {N_k}")
    s = get_s(N_k, acc)
    print(s)


def card_num(class_name:str) -> int:    
    if len(class_name) == 2: # 1-9
        card_num = int(class_name[1])
    else:  # 10, 11, 13
        card_num = int(class_name[1:])
    return card_num - 1


def class2label(class_name: str) -> int:
    # e.g) 3A*.jpg
    # 13s -> 0

    def shape2num(sh):
        if sh == "s":
            return 0
        elif sh == "h":
            return 1
        elif sh == "d":
            return 2
        elif sh == "c":
            return 3
        else:
            print(f"Unexpected shape: {sh}")
            sys.exit(1)

    card_shape: str
    if len(class_name) == 2: # 1-9
        card_shape = shape2num(class_name[0])
        card_num = 13 - int(class_name[1])
    else:  # 10, 11, 13
        card_shape = shape2num(class_name[0])
        card_num = 13 - int(class_name[1:])

    return card_shape * 13 + card_num

def get_s(N_k: List[int], acc: float) -> float:
    s = 0
    for i in range(13):
        x = N_k[i] * (i + 1)
        if x == 0:
            continue
        s += N_k[i] * math.log(x)
    return acc * s

def test():
    assert(class2label("s13") == 0)
    assert(class2label("s6") == 7)
    assert(class2label("s2") == 11)
    assert(class2label("h6") == 20)
    assert(class2label("c10") == 42)
    s = get_s([0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1.0)
    assert(round(s, 3) == 11.901)


if __name__ == "__main__":
    test()
    main()
