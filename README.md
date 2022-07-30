# CIL Road Segmentation Project
## Kaggle Team: WeNeedANameHere
### Team Members:
* Gianluca Danieletto
* Guillaume Dugat
* Felix Yang
* Daiwei Zhang

This repository contains the code of the different approaches we used for project 3 of the course *Computational Intelligence Lab*.

## Final Predictions:
To reproduce the result from our best approach (Res-U-Net-34, with external data and enhanced inference, described in section II.F and III.F in our report), please carefully follow the below instructions to perform the inference:

Please download [data.zip](https://drive.google.com/file/d/1jaS_45Bzl9lYbJIZk8_In0Ptu9Mf9p8Q/view?usp=sharing) and [experiment.zip](https://drive.google.com/file/d/1FyP_HDq0qAO2Tuekr8AZWLyOIDuDrMc7/view?usp=sharing).
Unzip and move the two folders under `final_approach`.

```bash

cd final_approach
cd src
pip install -r requirements.txt
python predict.py
```

will run the inference on all the images with the path
`data/test/images/*.png`, which takes around 2 minutes on a NVIDIA RTX 3080 GPU. 
All predicted binary maps will be written under the folder `data/test/prediction/`;
the `submission.csv` that produces the score of 0.91648 will be written under the `final_approach` home directory.
Note that a Pytorch installation with GPU/CUDA support is **required** for the prediction.

### File Overview
* `src/predict.py`: runs the enhanced inference with the trained model stored in `experiment/[experiment ID]/model.pth`
* `src/train.py`: train the model with external dataset (MA and DG). Please contact me if you need to run this script, since the post-processed external datasets are too large to include in our submission.
* `src/configuration.py`: contains hyper-parameter and configuration setting.
* `experiment/1658858769/model.pth`: our trained model.
* `experiment/1658858769/config.json`: stores the hyper-parameters and configurations we used to train our best model.
* `data/test/images/`: contains the 144 original test images. Feel free to replace them with any aerial images in png format.
* `data/test/prediction/`: running `predict.py` will generate 144 segmentation results in this directory. Make sure this directory exists before running the inference.
* `external_data/`: this folder contain 5 sample images and the corresponding segmentations each for MA and DG datasets.
* `submission.py`: our final submission with score 0.91648


## Notebooks:
**To run the notebook from our other approaches, place the training and test folders of the dataset into the same directory as the Python notebooks.**

### Notes:
* To keep the notebooks as clean, minimal and comprehensible as possible, we put almost all of the code into the `src_notebooks` directory.
* We deleted some output cells that contained many images to keep the file sizes of the notebooks small (e.g. output cells that show validation samples after each training epoch).
* For completeness, we added the code for the retrieval of additional data using the Google Maps API (`src_notebooks/google_maps_download.py`). However, to protect our API key, we ask the reader to download the pre-downloaded images from the Google Drive link as described in the `google_maps_processing.ipynb` notebook *(not relevant for our final approach)*.
* If image normalization is applied as a preprocessing step, the validation images shown after each epoch will look weird due to clipping since the normalized values are outside the valid range of pixel values.

### Overview:
* `augmentation_experiments.ipynb`: This notebook visualizes the dataset augmentation methods we use in other notebooks, such as flips, rotations, reflection paddings and color jitters. The main purpose of this notebook is to experiment with different parameters.
* `patch_cnn_baseline.ipynb`: We evaluate the Patch CNN baseline model taken from tutorial 10, and show whether it gets better or not with some basic strategies.
* `unet_baseline.ipynb`: In this notebook, we take the U-Net baseline model from tutorial 10 as a starting point and build upon it by introducing and evaluating one improvement after the other.
* `resnet_unet.ipynb`: We conduct experiments on a U-Net model where the down-sampling encoder is replaced by different ResNet models.
* `class_specific_models.ipynb`: In this approach, we use image embeddings and K-means clustering to assign implicit class labels to each sample of the given dataset. We then train a separate model for each implicit class.
* `google_maps_processing.ipynb`: This notebook generates many new samples from the downloaded Google Maps data by slicing the large images into smaller ones and applying rotations and flips.
* `large_dataset_training.ipynb`: Here we use additional Google Maps data to train our models with much larger datasets. We first train the model on our new dataset and then finetune it with the original dataset. We also apply morphological operations for post-processing.
* `dilation.ipynb`: We experiment the use of dilation in the bottleneck part of the models. We use two similar approaches.