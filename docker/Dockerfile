FROM tensorflow/tensorflow:2.7.0-gpu

RUN apt-get update
RUN apt-get install curl
RUN apt-get install python3-pip
RUN pip3 install virtualenv
RUN apt -y install git

RUN mkdir -p /eml-tools
COPY ./bootstrap.sh /eml-tools
WORKDIR /eml-tools
RUN chmod +x *.sh
RUN ./bootstrap.sh

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility