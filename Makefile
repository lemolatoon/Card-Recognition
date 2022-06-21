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
EPOCHS=150
YOLO_MODEL=m
N_TRAIN_EXP=36
DATA=card.yaml
#n, s, m, l, x
train_yolo:
	# python3 yolov5/train.py --img $(IMG_SIZE) --batch $(BATCH) --epochs $(EPOCHS) --data card.yaml --weights yolov5$(YOLO_MODEL).pt
	python3 yolov5/train.py --img $(IMG_SIZE) --batch $(BATCH) --epochs $(EPOCHS) --data $(DATA) --weights yolov5/runs/train/exp$(N_TRAIN_EXP)/weights/last.pt

# exp19: BATCH 128, epochs 64 IMG_SIZE 128 model: m with fake
# exp24: BATCH 16, epochs 209 IMG_SIZE 512 model: l with fake
# exp26: BATCH 16, epochs 30 IMG_SIZE 512 model: s witb real
# exp30: BATCH 16, epochs 50 IMG_SIZE 512 model: s  with real
# exp31: BATCH 16, epochs 100+exp30 IMG_SIZE 512 model: s with real
# exp32: BATCH 256, epochs 32+exp19 IMG_SIZE 128 model: m with real
# exp34: BATCH 16, epochs 100 IMG_SIZE 512 model: m with real
# exp36: BATCH 16, epochs 200+exp34 IMG_SIZE 512 model: m with real
# exp39: BATCH 16, epochs 150+exp36 IMG_SIZE 512 model: m with real
N_EXP=39
test_real_yolo:
	python3 yolov5/detect.py --img $(IMG_SIZE) --weights yolov5/runs/train/exp$(N_EXP)/weights/last.pt --source images/competition_sample/jpg 

SRC=trump.mp4
test_yolo:
	python3 yolov5/detect.py --img $(IMG_SIZE) --weights yolov5/runs/train/exp$(N_EXP)/weights/last.pt --source $(SRC)

rm_yolos:
	rm ~/workspace/Card-Recognition/images/datasets/yolo -r

all: setup cardimg datasets background train
