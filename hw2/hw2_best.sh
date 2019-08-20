# TODO: create shell script for running your improved model
wget https://www.dropbox.com/s/37a5fjeti2n7m8e/model_19_59.pth?dl=1 -O best.pth
python3 best.py $1 $2
