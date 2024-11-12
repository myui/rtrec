#!/bin/sh

ARCH=$(uname -m)
PROJECT_DIR=$(cd $(dirname $0); cd ..; pwd)

docker run --platform=linux/${ARCH} -it --rm -p 8888:8888 -p 6006:6006 -v ${PROJECT_DIR}:/home/td-user/rtrec myui/rtrec:${ARCH} \
bash -c "cd rtrec && source ~/.cargo/env && ~/.rye/shims/rye sync && ~/.rye/shims/jupyter-lab --ip=0.0.0.0 --no-browser --notebook-dir=/home/td-user/rtrec/notebooks"
