#!/bin/sh

ARCH=$(uname -m)
#ARCH=amd64
PROJECT_DIR=$(cd $(dirname $0); cd ..; pwd)

docker run --platform=linux/${ARCH} -it --rm -p 8888:8888 -p 6006:6006 -v ${PROJECT_DIR}/notebooks:/home/td-user/rtrec/notebooks myui/rtrec:${ARCH} \
bash -c "cd rtrec && git fetch && git pull && . ~/.local/bin/env && uv sync && uv run --with jupyter jupyter lab --ip=0.0.0.0 --no-browser --notebook-dir=/home/td-user/rtrec/notebooks"
