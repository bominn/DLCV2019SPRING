# TODO: create shell script for running your YoloV1-vgg16bn model
wget https://www.dropbox.com/s/2u7bbmikhn0cbur/yolo_38.pth?dl=1 -O yolo.pth
python3 test.py $1 $2
