# Based on https://github.com/ContinuumIO/docker-images/blob/master/anaconda3/Dockerfile
# With inspiration from https://github.com/wiseio/datascience-docker/blob/master/datascience-base/Dockerfile
FROM kcarnold/clubdl-base

MAINTAINER Kenneth C. Arnold <kcarnold@alum.mit.edu>

CMD [ "./run-notebook-server" ]
