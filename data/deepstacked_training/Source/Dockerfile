FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y sudo git curl build-essential lua5.1 luarocks

RUN luarocks install torch && \
    luarocks install nn && \
    luarocks install paths

WORKDIR /workspace