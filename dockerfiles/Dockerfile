#
## Build Docker image
#   $ docker build --platform=linux/amd64 -f ./Dockerfile -t myui/rtrec:0.1 .
#
## To run Jupyter notebook
#   $ docker run --platform=linux/amd64 -it --rm -p 8888:8888 -p 6006:6006 -v /Users/myui/workspace/myui/rtrec:/home/td-user/rtrec myui/rtrec:0.1 jupyter-lab --ip=0.0.0.0 --no-browser --notebook-dir=/home/td-user/rtrec/notebooks
#
## Run docker image
#   $ docker run --platform=linux/amd64 -it --rm myui/rtrec:0.1 bash
#

FROM python:3.12-slim-bookworm
# FROM python:3.9-slim-bookworm

ENV PATH="/home/td-user/.local/bin:$PATH"
ENV PYTHONPATH="/home/td-user/.local/lib/python3.12/site-packages"
ENV OPENBLAS_NUM_THREADS=1
ENV DEBCONF_NOWARNINGS=yes
# Enable fault handlers for the SIGSEGV, SIGFPE, SIGABRT, SIGBUS, and SIGILL signals.
# https://docs.python.org/3/library/faulthandler.html
ENV PYTHONFAULTHANDLER=1

RUN apt-get -y update \
 && apt-get install -y build-essential python3-dev git sudo wget curl \
 && apt-get install -y cmake libboost-all-dev libeigen3-dev libgsl0-dev \
 && apt-get install -y libopenblas-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

ADD requirements.txt /tmp/requirements.txt
ADD constraints.txt /tmp/constraints.txt

RUN pip install  --no-cache-dir --upgrade Cython \
 && pip install --no-cache-dir -c /tmp/constraints.txt -r /tmp/requirements.txt

RUN groupadd -g 65432 wheel \
 && useradd -u 65432 -g wheel --create-home -s /sbin/nologin td-user \
 && echo '%wheel ALL=NOPASSWD: /usr/local/bin/pip' >> /etc/sudoers \
 && echo '%wheel ALL=NOPASSWD: /usr/bin/apt-get' >> /etc/sudoers

USER td-user

WORKDIR /home/td-user

CMD ["python3"]