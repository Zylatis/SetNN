sudo chown -R graeme_gossel ~/.
#~ export LC_ALL="en_US.UTF-8"
#~ export LC_CTYPE="en_US.UTF-8"
#~ sudo dpkg-reconfigure locales

apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python
pip install imgaug
pip install -U pytest


