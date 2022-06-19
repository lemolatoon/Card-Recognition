.PHONY: setup cardimg datasets background train all
setup:
	sudo apt install poppler-utils
	pip install -r requirements.txt

cardimg:  pdf2img/pdf/0-24.pdf pdf2img/pdf/25-49.pdf pdf2img/pdf/50-51.pdf
	python3 pdf2img/main.py

datasets: 
	python3 make_dataset/main.py

background:
	python3 background/main.py

train:
	python3 cnn/main.py

IMG_SIZE=512
BATCH=16
EPOCHS=30
YOLO_MODEL=s
#n, s, m, l, x
train_yolo:
	python3 yolov5/train.py --img $(IMG_SIZE) --batch $(BATCH) --epochs $(EPOCHS) --data card.yaml --weights yolov5$(YOLO_MODEL).pt

# exp19: BATCH 128, epochs 64 IMG_SIZE 128 model: m
# exp24: BATCH 16, epochs 209 IMG_SIZE 512 model: l 
test_real_yolo:
	python3 yolov5/detect.py --img $(IMG_SIZE) --weights yolov5/runs/train/exp24/weights/last.pt --source images/competition_sample/jpg 

SRC=trump.mp4
test_yolo:
	python3 yolov5/detect.py --img $(IMG_SIZE) --weights yolov5/runs/train/exp24/weights/last.pt --source $(SRC)

rm_yolos:
	rm ~/workspace/Card-Recognition/images/datasets/yolo -r

all: setup cardimg datasets background train
