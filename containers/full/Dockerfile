FROM kcarnold/clubdl-base

RUN conda install -y \
    theano
RUN pip install keras
RUN pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp35-cp35m-linux_x86_64.whl
RUN mkdir ~/code

EXPOSE 8888
