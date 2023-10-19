#FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04 
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04

ARG REPO_NAME=${REPO_NAME}
ENV REPO_NAME=${REPO_NAME}

ARG DEBIAN_FRONTEND=noninteractive

# To save you a headache
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

ENV PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Fix Nvidia/Cuda repository key rotation
RUN sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list.d/*
RUN sed -i '/developer\.download\.nvidia\.com\/compute\/machine-learning\/repos/d' /etc/apt/sources.list.d/*  
RUN apt-key del 7fa2af80 &&\
	apt-get update && \
	apt-get  install -y wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb 

# CV2 Deps
RUN apt-get install -y ffmpeg libsm6 libxext6 libgl1

RUN apt update && apt install -y software-properties-common

# Install System Dependencies
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update

RUN apt-get install -y python3.9 \
    python3-pip \
    python3.9-venv \
    python3.9-dev \
    python3.9-distutils \
    curl \
    vim \
    git

# Adjust default python3 version to required version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
# Update pip3 version
RUN python3 -m pip install --upgrade pip

# poetry:
ENV POETRY_VERSION=1.1.14 \
  POETRY_NO_INTERACTION=1 \
  POETRY_VIRTUALENVS_CREATE=false \
  POETRY_CACHE_DIR='/var/cache/pypoetry' \
  POETRY_HOME='/usr/local'

# Install poetry
RUN curl -sSL 'https://install.python-poetry.org' | python3 -\
	&& poetry config virtualenvs.create false \
	&& poetry --version \
  	# Cleaning cache:
  	&& apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
  	&& apt-get clean -y && rm -rf /var/lib/apt/lists/*

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Create working directory 
WORKDIR /${REPO_NAME}
COPY . .

RUN --mount=type=cache,target=/root/.cache/poetry make env-docker

# TODO: Remove pip install and setup poetry - uncomment top lines
#RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

RUN git config --global --add safe.directory /${REPO_NAME}

#CMD ["python3", "./main.py"]
CMD ["python3", "sample_main/main.py"]