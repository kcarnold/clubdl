FROM kcarnold/clubdl-base

RUN apt-get install -y g++
RUN pip install theano && \
    pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp35-cp35m-linux_x86_64.whl && \
    pip install keras

EXPOSE 8888
