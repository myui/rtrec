#!/bin/sh

ARCH=`uname -m`
echo "Building docker image for ${ARCH}..."

PROJECT_DIR=$(cd $(dirname $0); cd ..; pwd)
cd ${PROJECT_DIR}

docker buildx build --platform=linux/${ARCH} -f ./dockerfiles/Dockerfile --load -t myui/rtrec:${ARCH} .
echo "Docker image 'myui/rtrec:${ARCH}' built successfully!"
