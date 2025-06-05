#!/bin/bash
ROMAN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $ROMAN_DIR

# Install CLIPPER
git submodule update --init --recursive
mkdir dependencies/clipper/build
cd dependencies/clipper/build
cmake .. && make && make pip-install

# Install Kimera-RPGO
mkdir $ROMAN_DIR/dependencies/Kimera-RPGO/build
cd $ROMAN_DIR/dependencies/Kimera-RPGO/build
cmake .. && make

# pip install
cd $ROMAN_DIR
pip install --retries=3 --default-timeout=1000 .

# download weights
mkdir -p $ROMAN_DIR/weights
cd $ROMAN_DIR/weights
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
gdown 'https://drive.google.com/uc?id=1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv'