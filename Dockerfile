#FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
#FROM ubuntu:18.04
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub 
#COPY ./cuda-keyring_1.0-1_all.deb cuda-keyring_1.0-1_all.deb

#RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list
#RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update
# Setup Environment Variables
ENV DEBIAN_FRONTEND="noninteractive" \
    TZ="Europe/London"

ENV COSMOS_BASE_DIR="/opt/COSMOS" \
    COSMOS_DATA_DIR="/mmsys22cheapfakes" 
#    COSMOS_IOU="0.25" \
#    COSMOS_RECT_OPTIM="1"

# Copy Dependencies
COPY requirements.txt /
COPY detectron2 /detectron2

#COPY detectron2_changes /detectron2_changes
RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*
# Prepare Environment (1)
RUN mkdir -p /opt/COSMOS/models_final

#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
#RUN dpkg -i cuda-keyring_1.0-1_all.deb
# Install Python
RUN apt-get update && apt-get install -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    #add-apt-repository -y ppa:deadsnakes/ppa && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.8 python3-dev python3-pip python3-opencv \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Prepare Python Dependencies
RUN python3 -m pip install --upgrade pip && \
    pip3 install cython numpy setuptools && \
    pip3 install -r /requirements.txt

# Patch and Install Detectron
#RUN cd /detectron2/ && \
#    patch -p1 < /detectron2_changes/0001-detectron2-mod.patch && \
#    cd / && python3 -m pip install -e detectron2
RUN python3 -m pip install -e detectron2
# Fix PyCocoTools
RUN pip3 uninstall -y pycocotools && \
    pip3 install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
# Download spaCy
RUN python3 -m spacy download en && \
    python3 -m spacy download en_core_web_sm

# Copy Source
COPY run_cheapfake.py /
COPY . /opt/COSMOS
COPY start.sh /opt/COSMOS
COPY utils /opt/COSMOS/utils/
COPY model_archs /opt/COSMOS/model_archs/
COPY newresults /opt/COSMOS/newresults/
COPY models_final /opt/COSMOS/models_final/
COPY test_data.json /opt/COSMOS
COPY mmsys22cheapfakes /mmsys22cheapfakes/
COPY mmsys22cheapfakes /opt/COSMOS/mmsys22cheapfakes/
#RUN python3 -m pip uninstall numpy
#RUN python3 -m pip install numpy-indexed
#RUN python3 -m pip install --upgrade pip
#RUN python3 -m pip uninstall -y numpy
#RUN python3 -m pip install -U numpy
#RUN python3 -m pip uninstall -y opencv-python
#RUN python3 -m pip install -U opencv-python

#RUN python3 -m pip install numpy --upgrade --ignore-installed
# Start the code
ENTRYPOINT []
#CMD [ "python3", "./run_cheapfake.py" ]
CMD ["/opt/COSMOS/start.sh"]