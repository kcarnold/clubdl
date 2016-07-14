#!/bin/sh
set -e
docker build -t kcarnold/clubdl-base -f Dockerfile.base .
docker build -t kcarnold/clubdl -f Dockerfile .
