import torch
import cv2
import os
import sys
sys.path.append(os.pardir)

# --- 検出する際のモデルを読込 ---
# model = torch.hub.load('ultralytics/yolov5','yolov5s')#--- webのyolov5sを使用
model = torch.hub.load("../yolov5", 'yolov5s', source='local')  # --- localのyolov5sを使用


# --- 検出の設定 ---
model.conf = 0.5  # --- 検出の下限値（<1）。設定しなければすべて検出
model.classes = [0]  # --- 0:person クラスだけ検出する。設定しなければすべて検出
# print(model.names) #--- （参考）クラスの一覧をコンソールに表示

# --- 映像の読込元指定 ---
# camera = cv2.VideoCapture("../pytorch_yolov3/data/sample.avi")#--- localの動画ファイルを指定
camera = cv2.VideoCapture(0)  # --- カメラ：Ch.(ここでは0)を指定


# --- 画像のこの位置より左で検出したら、ヒットとするヒットエリアのためのパラメータ ---
pos_x = 240

while True:

    # --- 画像の取得 ---
    #  imgs = 'https://ultralytics.com/images/bus.jpg'#--- webのイメージファイルを画像として取得
    #  imgs = ["../pytorch_yolov3/data/dog.png"] #--- localのイメージファイルを画像として取得
    ret, imgs = camera.read()  # --- 映像から１フレームを画像として取得

# --- 推定の検出結果を取得 ---
#  results = model(imgs) #--- サイズを指定しない場合は640ピクセルの画像にして処理
    results = model(imgs, size=160)  # --- 160ピクセルの画像にして処理

# --- 出力 ---
# --- 検出結果を画像に描画して表示 ---
    #--- 各検出について
    for *box, conf, cls in results.xyxy[0]:  # xyxy, confidence, class

        #--- クラス名と信頼度を文字列変数に代入
        s = model.names[int(cls)] + ":" + '{:.1f}'.format(float(conf) * 100)

        # --- ヒットしたかどうかで枠色（cc）と文字色（cc2）の指定
        if int(box[0]) > pos_x:
            cc = (255, 255, 0)
            cc2 = (128, 0, 0)
        else:
            cc = (0, 255, 255)
            cc2 = (0, 128, 128)

        #--- 枠描画
        cv2.rectangle(
            imgs,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color=cc,
            thickness=2,
        )
        #--- 文字枠と文字列描画
        cv2.rectangle(imgs, (int(box[0]), int(box[1]) - 20), (int(box[0]) + len(s) * 10, int(box[1])), cc, -1)
        cv2.putText(imgs, s, (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_PLAIN, 1, cc2, 1, cv2.LINE_AA)

    #--- ヒットエリアのラインを描画
    cv2.line(imgs, (pos_x, 0), (pos_x, 640), (128, 128, 128), 3)

    #--- 描画した画像を表示
    cv2.imshow('color', imgs)

# --- （参考）yolo標準機能を使った出力 ---
#  results.show()#--- yolo標準の画面表示
#  results.print()#--- yolo標準のコンソール表示

# --- （参考）yolo標準の画面を画像取得してopencvで表示 ---
#  pics = results.render()
#  pic = pics[0]
#  cv2.imshow('color',pic)

    # --- 「q」キー操作があればwhileループを抜ける ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
