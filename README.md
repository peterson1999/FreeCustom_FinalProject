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

### To activate cosine based scheduler for dynamic weight scaling
Go to `config_stable_diffusion.yaml` and set `use_cosine_scheduler` to `True`. If `use_cosine_scheduler` is not in the config file by default, you can add it.

### To use Auxiliary Concept Information-Based Dynamic Mask Weight Scaling
```
cd FreeCustom
python 5.2_freecustom.py
```
### When using one's own dataset for style transfer
Be sure to have the style reference image at the very last in `ref_image_infos:` in `config_stable_diffusion.py`

### When creating your own data using Data Processing Pipeline
- Run Resize_Image_and_Mask.ipynb in FreeCustom folder
- Upload your photos to data folder
- Draw your bounding boxes
- Resized images should be saved at data/resized and masked images should be saved at data/masked
- Transfer images and masks to dataset folder in freecustom, name folder to target concept of images (i.e, “man_hat_sunglasses”)
- Edit configuration.yaml, refer to existing data samples

### 🌟 Customization
We prepare several test data across various styles in `./dataset` for you to reproduce the results in our paper. 

You only need to replace `./configs/config_stable_diffusion.yaml` with one of configuration files in the `./dataset/` folder,
and run the previous code again

