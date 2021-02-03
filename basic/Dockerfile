# Im using nvidia base image. to use CPU image, uncomment the below line###
FROM ubuntu:18.04
#FROM nvidia/cuda:11.0-base

MAINTAINER Jashwant Raj (github.com/jashwantraj92)
ARG DEBIAN_FRONTEND=noninteractive
#### installing python and known dependencies
RUN apt-get update && \
    apt-get install -y \
        python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN apt-get -y update && apt-get install libgl1-mesa-glx ffmpeg libsm6 libxext6  -y

#### installing required python packages###
RUN pip3 install --upgrade \
        pip && \
        pip3 install tensorflow keras pillow image matplotlib numpy opencv-python

#### copying the traning/prediction python script
RUN mkdir /home/mnist-learning
COPY keras-mnist.py /home/mnist-learning

WORKDIR /home/mnist-learning
VOLUME /mnt/data/mnist:/mnt/data/mnist
ENTRYPOINT ["python3","keras-mnist.py"]
#ENTRYPOINT [ "python3","predict.py" ]
