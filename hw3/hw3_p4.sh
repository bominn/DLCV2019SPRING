# TODO: create shell script for running your improved UDA model
if [ ! -f ./model/mnistm2svhn_vae ]; then
	wget https://www.dropbox.com/s/4jdqzsx86iowda4/mnistm2svhn_vae?dl=1 -O model/mnistm2svhn_vae
fi

if [ ! -f ./model/svhn2usps_vae ]; then
	wget https://www.dropbox.com/s/2pbtzpybwb7u177/svhn2usps_vae?dl=1 -O model/svhn2usps_vae
fi

if [ ! -f ./model/usps2mnistm_vae ]; then
	wget https://www.dropbox.com/s/t1iqln1raoxjx7a/usps2mnistm_vae?dl=1 -O model/usps2mnistm_vae
fi

python3 p4.py $1 $2 $3
