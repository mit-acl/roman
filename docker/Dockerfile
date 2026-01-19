FROM ubuntu:22.04

# Avoid interactive tzdata configuration
ENV DEBIAN_FRONTEND=noninteractive

# Essentials
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -q -y --no-install-recommends apt-utils
RUN apt-get install -q -y --no-install-recommends net-tools iproute2 iputils-ping
RUN apt-get install -q -y --no-install-recommends git tmux vim neovim tmuxp curl wget
RUN apt-get install -q -y --no-install-recommends python3 python3-pip python3-venv python3-dev
RUN apt-get install -q -y --no-install-recommends cmake g++ make
# required for CLIPPER
RUN apt-get install -q -y --no-install-recommends libeigen3-dev
RUN apt-get install -q -y --no-install-recommends libblas-dev liblapack-dev
# required for OpenCV
RUN apt-get install -q -y --no-install-recommends ffmpeg libsm6
# required for Kimera-RPGO
RUN apt-get install -q -y --no-install-recommends libboost-all-dev

WORKDIR /root
RUN git clone https://github.com/mit-acl/roman.git roman

# Install gtsam for Kimera-RPGO
RUN git clone https://github.com/borglab/gtsam.git ./roman/dependencies/gtsam
WORKDIR /root/roman/dependencies/gtsam
RUN git checkout 686e16aaae26c9a4f23d4af7f2d4a504125ec9c3
# TODO - use multiple cores
RUN mkdir build 
WORKDIR /root/roman/dependencies/gtsam/build
RUN cmake .. -DGTSAM_POSE3_EXPMAP=ON -DGTSAM_ROT3_EXPMAP=ON
RUN make install && ldconfig

# Install ROMAN
WORKDIR /root
RUN ./roman/install.sh
