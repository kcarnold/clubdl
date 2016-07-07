# Based on https://github.com/ContinuumIO/docker-images/blob/master/anaconda3/Dockerfile
# With inspiration from https://github.com/wiseio/datascience-docker/blob/master/datascience-base/Dockerfile
FROM krallin/ubuntu-tini:14.04

MAINTAINER Kenneth C. Arnold <kcarnold@alum.mit.edu>

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update --fix-missing && apt-get install -y \
    bzip2 \
    curl \
    ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git \
    g++ \
    wget

RUN echo 'export PATH=/anaconda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-4.1.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /anaconda && \
    rm ~/anaconda.sh

ENV PATH /anaconda/bin:$PATH

RUN pip install theano && \
    pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp35-cp35m-linux_x86_64.whl && \
    pip install keras

EXPOSE 8888
VOLUME /workspace
WORKDIR /workspace
CMD [ "/bin/bash" ]
