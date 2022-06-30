# Card-Recognition
## 準備
```bash
mkdir -p images/cards
mkdir -p images/backgrounds
```
## yoloを用いた学習
### データセットを`images/datasets/yolo`配下におく
ファイル構造の例
```
|
|
|
yolo---root---|--images
              |
              |--labels
```

### データセットをtrainとvalに分ける
`make_dataset/split.py`の変数`dir_name`をデータセットのディレクトリ名に変更する。上の例ならば`root`に変更する。<br>
そして、`python3 make_dataset/main.py`

### パラメータを調整する
`Makefile`の`EPOCH`や`BATCH`,`YOLO_MODEL`を変更する。`YOLO_MODEL`はモデルの大きさを表していて、n, s, m, l, xから選べる。`IMG_SIZE`は画像の解像度である。大きいほど学習時間が長くなる。精度がどのくらい良くなるのかは不明。

### 学習させる
`make train_yolo`を実行する。`make`は`apt install make`でいれよう。

### 学習結果を吟味する
学習が終わると、`yolov5/runs/train/`の下に`exp`から始まるディレクトリができている。このsuffixの最も大きいディレクトリが最新の学習結果である。
たとえば、`exp10`が最も大きいならば、`yolov5/runs/train/exp10/weights`の中に学習後の重みが入っている。もう一度`Makefile`を開き、test_yoloの方のpathを最新のexpに変更し、SRC変数に画像ファイルや写真が入ったフォルダを指定すれば、`yolov5/runs/detect/`以下に生成される。

## competiの使い方
- `competi/images/`以下に`*.jpg`と`label.txt`を配置する
- `run.sh`を実行すると、Sが計算されている！！！