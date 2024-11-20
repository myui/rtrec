#!/bin/sh

#ARCH=`uname -m`
#echo "Building docker image for ${ARCH}..."

PROJECT_DIR=$(cd $(dirname $0); cd ..; pwd)
cd ${PROJECT_DIR}/dockerfiles

docker buildx build --platform=linux/amd64,linux/arm64 -f ./Dockerfile .
docker buildx build --load --platform linux/amd64 -t myui/rtrec:amd64 .
docker buildx build --load --platform linux/arm64 -t myui/rtrec:arm64 .

echo "Docker images 'myui/rtrec:arm64' and 'myui/rtrec:amd64' built successfully!"
