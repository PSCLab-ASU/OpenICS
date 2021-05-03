FROM gpuci/miniconda-cuda:10.1-runtime-ubuntu18.04

RUN mkdir /code

COPY ./ /code/

COPY ./environment/OpenICS_Pytorch.yml /
COPY ./environment/OpenICS_Tensorflow.yml /
COPY ./environment/postInstall /

RUN chmod +x /postInstall
RUN /postInstall
RUN apt-get update
RUN apt-get nano

WORKDIR /code
