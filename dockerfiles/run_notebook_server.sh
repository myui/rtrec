#!/bin/sh

docker run --platform=linux/amd64 -it --rm -p 8888:8888 -p 6006:6006 -v /Users/myui/workspace/myui/rtrec:/home/td-user/rtrec myui/rtrec:0.1 jupyter-lab --ip=0.0.0.0 --no-browser --notebook-dir=/home/td-user/rtrec/notebooks
