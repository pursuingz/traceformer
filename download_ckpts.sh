#!/bin/bash
mkdir -p checkpoints
pip install -U "huggingface_hub[cli]"
huggingface-cli download chenwang/physctrl --local-dir checkpoints --local-dir-use-symlinks False