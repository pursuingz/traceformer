<h1 align="center">PhysCtrl: Generative Physics for Controllable and Physics-Grounded Video Generation  </h1>
<p align="center"><a href="https://arxiv.org/abs/2509.20358"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://cwchenwang.github.io/physctrl/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/chenwang/physctrl'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live_Demo-blue'></a>
</p>


## 📦 Installation

```bash
python3.10 -m venv physctrl
source physctrl/bin/activate
# CAUTION: change it to your CUDA version
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118 xformers
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu118.html
pip install -r requirements.txt
```



## 🤖 Pretrained Models

Download checkpoints:
```bash
bash download_ckpts.sh
```



## 📂 Dataset

Currently, it's difficult for us to release full dataset since it's too large. But since our dataset is based on the open-source [TRELLIS-500K](https://github.com/microsoft/TRELLIS/blob/main/DATASET.md), it would be easy to recreate our dataset. Here we provide the scripts for creating the dataset for elastic, plasticine and sand material.

1. Download the Objaverse sketchfab dataset

   ``` bash
   cd src/data_generation
   python3 dataset_toolkits/build_metadata.py ObjaverseXL --source sketchfab --output_dir data/objaverse
   python3 dataset_toolkits/download.py ObjaverseXL --output_dir data/objaverse
   ```

2. Generate **h5** data with MPM simulator for different materials

   ```bash
   # Use "--uid_list configs/objaverse_valid_uid_list.json" to include the full dataset
   python3 generate_mpm_data.py	--material elastic --start_idx 0 --end_idx 1 --visualization 
   python3 generate_mpm_data.py	--material plasticine --start_idx 0 --end_idx 1 --visualization
   python3 generate_mpm_data.py	--material sand --start_idx 0 --end_idx 1 --visualization
   ```

   You can view the simulated trajectories in `src/data_generation/data/objaverse/visualization`

   


## 🎥 Image to Video Generation
We provide several examples in the `examples` folder. You can put your own example there using the same format.
```bash
cd src
python3 inference.py --data_name "penguin"
```



## 🏋️‍♂️ Training and Evaluation

### Inference Trajectory Generation
```bash
python3 eval.py --config configs/eval_base.yaml
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

