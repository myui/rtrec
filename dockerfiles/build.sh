#!/bin/sh

docker buildx build --platform=linux/amd64 -f ./Dockerfile --load -t myui/rtrec:0.1 .