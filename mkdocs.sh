#!/usr/bin/env bash

# docker build -t mkdocs .

# Available commands:
# serve: local doc
# gh-deploy: deploy to github
docker run --rm -it -v ~/.ssh:/root/.ssh -v ${PWD}:/docs -p 8000:8000 mkdocs $1
