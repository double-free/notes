#!/usr/bin/env bash

docker build -t mkdocs .

# Available commands:
# serve: local doc
# gh-deploy: deploy to github
docker run --rm -it -v ~/.ssh:/root/.ssh -v ${PWD}:/docs mkdocs $1
