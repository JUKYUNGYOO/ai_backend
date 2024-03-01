# install guide

$ git clone https://github.com/orion-sod/ai_back.git

# base model
install guild
https://detrex.readthedocs.io/en/latest/tutorials/Installation.html

$ git clone https://github.com/IDEA-Research/detrex.git
$ cd detrex
$ git submodule init
$ git submodule update

/ai_backend/detrex 위치 시키고

# docker compose build 
$ cd /ai_backend 
$ docker compose build test-dev
$ dcker compose up test-dev

docker start 254d9db7c507(containerID)
docker exec -it 254d9db7c507 /bin/bash

# detrex로 이동 후 설치. 
cd /workspace/ai_backend/detrex
$ pip install -e . 



