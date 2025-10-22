<h1 align="center">PhysCtrl: Generative Physics for Controllable and Physics-Grounded Video Generation  </h1>
<p align="center"><a href="https://arxiv.org/abs/2509.20358"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://cwchenwang.github.io/physctrl/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/chenwang/physctrl'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live_Demo-blue'></a>
</p>

**Still working on remaining parts...**

## 📦 Installation
```bash
python3.10 -m venv venv
source $venv/bin/activate
# change it to your CUDA version
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118 xformers
pip install -r requirements.txt
```

## 🤖 Pretrained Models
Checkpoints can be found at: https://huggingface.co/chenwang/physctrl/tree/main

## 📂 Dataset
TBD

## 🎥 Image to Video Generation
```bash
TBD
```

## 🏋️‍♂️ Training and Evaluation
### Inference Trajectory Generation
```bash
python3 eval.py --config configs/eval_v3.yaml
```

### Train Trajectory Generation
For base model (support elastic objects with different force directions, fast inference, works for most cases):
```bash
accelerate launch --config_file configs/acc/8gpu.yaml train.py --config configs/config_dit_base.yaml
```

For large model (support all elastic, plasticine, sand and rigid objects, the latter three only supports gravity as force):
```ba
accelerate launch --config_file configs/acc/8gpu.yaml train.py --config configs/config_dit_large.yaml
```

### Evaluate Trajectory Generation
```bash
python3 volume_iou.py --split_lst EVAL_DATASET_PATH --pred_path PRED_RESULTS_PATH
```

### Estimating Physical Parameters
```bash
python3 -m utils.physparam --config configs/eval_base.yaml
```

## 📜 Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@article{wang2024physctrl,
    title   = {PhysCtrl: Generative Physics for Controllable and Physics-Grounded Video Generation},
    author  = {Wang, Chen and Chen, Chuhao and Huang, Yiming and Dou, Zhiyang and Liu, Yuan and Gu, Jiatao and Liu, Lingjie},
    journal = {NeurIPS},
    year    = {2025}
}
```

