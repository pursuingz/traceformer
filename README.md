<h1 align="center">Traceformer  </h1>
</p>


## 📦 Installation

```bash
python3.10 -m venv traceformer
source physctrl/bin/activate
# CAUTION: change it to your CUDA version
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118 xformers
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu118.html --no-build-isolation
pip install git+https://github.com/ashawkey/diff-gaussian-rasterization.git --no-build-isolation
pip install -r requirements.txt
```



## 🤖 Pretrained Models

Download checkpoints:
```bash
bash download_ckpts.sh
```



## 📂 Dataset

Due to the large storage of original data, it's difficult for us to release the full dataset. A subset of the data can be found at [here](https://huggingface.co/datasets/chenwang/physctrl/resolve/main/sample.zip). Since our dataset is based on the open-source [TRELLIS-500K](https://github.com/microsoft/TRELLIS/blob/main/DATASET.md), it would be easy to recreate our dataset. Here we provide the scripts for creating the dataset for elastic, plasticine and sand material.

1. Download the Objaverse sketchfab dataset

   ``` bash
   cd src/data_generation
   python dataset_toolkits/build_metadata.py ObjaverseXL --source sketchfab --output_dir data/objaverse
   python dataset_toolkits/download.py ObjaverseXL --output_dir data/objaverse
   ```

2. Generate **h5** data with MPM simulator for different materials

   ```bash
   # Use "--uid_list configs/objaverse_valid_uid_list.json" to include the full dataset
   python generate_mpm_data.py	--material elastic --start_idx 0 --end_idx 1 --visualization 
   python generate_mpm_data.py	--material plasticine --start_idx 0 --end_idx 1 --visualization
   python generate_mpm_data.py	--material sand --start_idx 0 --end_idx 1 --visualization
   ```

   You can view the simulated trajectories in `src/data_generation/data/objaverse/visualization`




## 🏋️‍♂️ Training and Evaluation

### Inference Trajectory Generation
```bash
python eval.py --config configs/eval_base.yaml
```

### Train Trajectory Generation
For base model (support elastic objects with different force directions, fast inference, works for most cases):
```bash
accelerate launch --config_file configs/acc/1gpu.yaml train.py --config configs/config_dit_base.yaml
```

For large model (support all elastic, plasticine, sand and rigid objects, the latter three only supports gravity as force):
```ba
accelerate launch --config_file configs/acc/1gpu.yaml train.py --config configs/config_dit_large.yaml
```

### Evaluate Trajectory Generation
```bash
python volume_iou.py --split_lst EVAL_DATASET_PATH --pred_path PRED_RESULTS_PATH
```

### Estimating Physical Parameters
```bash
python -m utils.physparam --config configs/eval_base.yaml
```


