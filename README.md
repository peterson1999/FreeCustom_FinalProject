# FreeCustom_FinalProject

## 🔨 Installation

```
conda create -n st_freecustom python=3.10 -y
conda activate st_freecustom
pip install -r requirements.txt
```
## 🚀 Run
### FreeCustom

```
cd FreeCustom
python freecustom_stable_diffusion.py
```

### StyleTransferFreeCustom
```
cd StyleTransferFreeCustom
python magicface_stable_diffusion.py
```

### 🌟 Customization
We prepare several test data across various styles in `./dataset` for you to reproduce the results in our paper. 

You only need to replace `./configs/config_stable_diffusion.yaml` with one of configuration files in the `./dataset/` folder,
and run the previous code again

