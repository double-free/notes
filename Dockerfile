FROM ghcr.io/afritzler/mkdocs-material
RUN sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories \
    && apk update -v
RUN apk add openssh && apk add git
