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

train_yolo:
	python3 yolov5/train.py --img 256 --batch 64 --epochs 64 --data card.yaml --weights yolov5s.pt

test_yolo:
	python3 yolov5/detect.py --img 256 --weights yolov5/runs/train/exp8/weights/last.pt --source images/competition_sample/jpg 

rm_yolos:
	rm ~/workspace/Card-Recognition/images/datasets/yolo -r

all: setup cardimg datasets background train
