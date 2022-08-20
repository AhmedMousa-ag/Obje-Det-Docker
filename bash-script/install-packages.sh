#!/bin/bash


echo "Updating apt"
apt update -y
echo "Finished updating apt"
python -m pip install -U pip
apt-get install git -y

pip install matplotlib
pip install pandas
pip install scipy
pip install imageio
pip install ipython

git clone --depth 1 https://github.com/tensorflow/models

echo "Installing protoc"
PROTOC_ZIP=protoc-3.15.8-linux-x86_64.zip

curl -OL https://github.com/google/protobuf/releases/download/v3.15.8/$PROTOC_ZIP
unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
unzip -o $PROTOC_ZIP -d /usr/local include/*
rm -f $PROTOC_ZIP
echo "Finished installing protoc"

cd /app/
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .

cd /app/





