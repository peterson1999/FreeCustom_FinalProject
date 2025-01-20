# Evaluation

Test image2image and image2text in the existing environment:

## image2image

A_images: Folder path containing generated images.

B_images: Folder path containing reference images.

### 1-to-1
Make sure the images are in the same order, ideally with matching filenames

run the following:
```
python image2image.py /path/to/A_images /path/to/B_images
```

### 1-to-many

For 1-to-many pairing, provide a mapping file mapping.txt with the following format:
```
A_1.jpg, B_1.jpg, B_2.jpg
A_2.jpg, B_3.jpg, B_4.jpg,  B_5.jpg
```

use the --mapping-file argument:
```
python image2image.py /path/to/A_images /path/to/B_images --mapping-file mapping.txt
```

## image2text
image_path: Path to the folder containing image files.

text_path: Path to the folder containing text files. 

Make sure the images and text files are in the same order, ideally with matching filenames

run the following:
```
python image2text.py /path/to/image /path/to/text
```

## iqa_score
Create a new environment:
```
conda create -n clipiqa python=3.9 -y
conda activate clipiqa
pip install -r iqa_req.txt
```

run the following:
```
python iqa_score.py path/to/img
```