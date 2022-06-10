.PHONY: setup cardimg datasets background train all
setup:
	sudo apt install poppler-utils
	pip install -r requirements.txt

cardimg:  pdf2img/pdf/0-24.pdf pdf2img/pdf/25-49.pdf pdf2img/pdf/50-51.pdf
	python3 pdf2img/main.py

datasets: 
	python3 make_dataset/main.py

background:
	python3 background/desk.py

train:
	python3 cnn/main.py

all: setup cardimg datasets background train
