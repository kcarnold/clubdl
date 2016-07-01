#!/bin/sh
docker-machine create -d virtualbox --virtualbox-disk-size "50000" --virtualbox-cpu-count "4" --virtualbox-memory "8092" docker-big
