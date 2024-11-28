#!/bin/sh

set -ev

ARCH=`uname -m`
echo "Building docker image for ${ARCH}..."

PROJECT_DIR=$(cd $(dirname $0); cd ..; pwd)
cd ${PROJECT_DIR}/dockerfiles

#docker buildx build --platform=linux/amd64,linux/arm64 -f ./Dockerfile .
#docker buildx build --load --platform linux/amd64 -t myui/rtrec:amd64 .
#docker buildx build --load --platform linux/arm64 -t myui/rtrec:arm64 .
docker buildx build --platform=linux/${ARCH} -f ./Dockerfile .  --load -t myui/rtrec:${ARCH}

echo "Docker images 'myui/rtrec:${ARCH}' built successfully!"