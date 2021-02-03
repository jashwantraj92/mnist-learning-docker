# Im using nvidia base image. to use CPU image, uncomment the below line###
#FROM ubuntu:18.04
FROM nvidia/cuda:11.0-base

MAINTAINER Jashwant Raj (github.com/jashwantraj92)
ARG DEBIAN_FRONTEND=noninteractive
#### installing python and known dependencies
RUN apt-get update && \
    apt-get install -y \
        python3-tk python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN apt-get -y update && apt-get install libgl1-mesa-glx ffmpeg libsm6 libxext6  -y

#### installing required python packages###
RUN pip3 install --upgrade \
        pip && \
        pip3 install tensorflow-gpu keras pillow image matplotlib numpy opencv-python

#### copying the traning/prediction python script
COPY mnist-learning /repositories/mnist-learning

WORKDIR /repositories/mnist-learning
VOLUME /mnt/data/Data/mnist:/mnt/data/Data/mnist
ENTRYPOINT ["python3","train.py"]
#ENTRYPOINT [ "python3","predict.py" ]
