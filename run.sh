#! /bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd $SCRIPT_DIR

# rm $SCRIPT_DIR/competi/out -r
# python3 $SCRIPT_DIR/yolov5/detect.py --img 512 --data $SCRIPT_DIR/card_fake.yaml --weights $SCRIPT_DIR/yolov5/runs/train/exp47/weights/last.pt --source $SCRIPT_DIR/competi/images --project $SCRIPT_DIR/competi/out --save-txt --save-conf
python3 competi/main.py