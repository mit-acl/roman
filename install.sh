#!/bin/bash
ROMAN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $ROMAN_DIR

# Install CLIPPER
git submodule update --init --recursive
mkdir -p dependencies/clipper/build
cd dependencies/clipper/build
cmake .. && make && make pip-install

# Install Kimera-RPGO
cd $ROMAN_DIR/dependencies
git clone git@github.com:MIT-SPARK/Kimera-RPGO.git 
cd Kimera-RPGO && git checkout d28b4df
mkdir -p $ROMAN_DIR/dependencies/Kimera-RPGO/build
cd $ROMAN_DIR/dependencies/Kimera-RPGO/build
cmake .. && make

# pip install
cd $ROMAN_DIR
pip install .