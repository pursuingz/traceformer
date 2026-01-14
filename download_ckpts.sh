#!/bin/bash
mkdir -p checkpoints
pip install -U "huggingface_hub[cli]<1.0"
huggingface-cli download chenwang/physctrl --local-dir checkpoints --local-dir-use-symlinks False
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoints/sam_vit_h_4b8939.pth
