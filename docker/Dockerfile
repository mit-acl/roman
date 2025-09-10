FROM ubuntu:22.04

# Avoid interactive tzdata configuration
ENV DEBIAN_FRONTEND=noninteractive

# Essentials
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -q -y --no-install-recommends apt-utils
RUN apt-get install -q -y --no-install-recommends net-tools iproute2 iputils-ping
RUN apt-get install -q -y --no-install-recommends git tmux vim neovim tmuxp curl
RUN apt-get install -q -y --no-install-recommends python3 python3-pip python3-venv python3-dev
RUN apt-get install -q -y --no-install-recommends cmake g++ make
# required for CLIPPER
RUN apt-get install -q -y --no-install-recommends libeigen3-dev
RUN apt-get install -q -y --no-install-recommends libblas-dev liblapack-dev
# required for OpenCV
RUN apt-get install -q -y --no-install-recommends ffmpeg libsm6

WORKDIR /root
RUN git clone https://github.com/mit-acl/roman.git roman
# RUN python3 -m venv .env
# RUN source .env/bin/activate
RUN ./roman/install.sh
# RUN pip install --upgrade pip
# RUN pip install wheel setuptools
# cd roman/dependencies/clipper/build/ && cmake .. && make
# python3 ~/roman/demo/demo.py -p ~/roman/demo/params/demo/ -o ~/demo_output