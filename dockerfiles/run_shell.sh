#!/bin/sh

ARCH=$(uname -m)
PROJECT_DIR=$(cd $(dirname $0); cd ..; pwd)

docker run --platform=linux/${ARCH} -it --rm -v ${PROJECT_DIR}:/home/td-user/rtrec myui/rtrec:${ARCH} \
bash -c "cd rtrec && source ~/.cargo/env && ~/.rye/shims/rye sync && ~/.rye/shims/rye remove rtrec && export RUST_LOG=debug && bash"
