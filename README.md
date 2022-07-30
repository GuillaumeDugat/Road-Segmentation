# CIL Road Segmentation Project
## Kaggle Team: WeNeedANameHere
### Team Members:
* Gianluca Danieletto
* Guillaume Dugat
* Felix Yang
* Daiwei Zhang

### Setup:
* TODO!!!
* Place the training and test folders of the dataset into the same directory as the Python notebooks.
* Go to the following link, download ***large.zip*** and place the unzipped folder into the same directory as the Python notebooks: https://drive.google.com/drive/folders/1GNzTsWcSkkLJfd1lcTLmzBma9KkU2oxH?usp=sharing

### Notes:
* To keep the notebooks as clean, minimal and comprehensible as possible, we put almost all of the code into the *src_notebooks* directory.
* We deleted some output cells that contained many images to keep the file sizes of the notebooks small (e.g. output cells that show validation samples after each training epoch or all predictions on the test set).
* For completeness, we added the code for the retrieval of additional data using the Google Maps API (*src/google_maps_download.py*). However, to protect our API key and money, we ask the reader to download the pre-downloaded images from the Google Drive link above.
* If image normalization is applied as a preprocessing step, the validation images shown after each epoch will look weird due to clipping since the normalized values are outside the valid range of pixel values.

### Final Predictions:
To reproduce the result from our best approach, TODO: @Daiwei

### Notebooks:
* **augmentation_experiments.ipynb**: This notebook visualizes the dataset augmentation methods we use in other notebooks, such as flips, rotations, reflection paddings and color jitters. The main purpose of this notebook is to experiment with different parameters.
* **patch_cnn_baseline.ipynb**: We evaluate the Patch CNN baseline model taken from tutorial 10, and show whether it gets better or not with some basical strategies.
* **unet_baseline.ipynb**: In this notebook, we take the U-Net baseline model from tutorial 10 as a starting point and build upon it by introducing and evaluating one improvement after the other.
* **resnet_unet.ipynb**: We conduct experiments on a U-Net model where the down-sampling encoder is replaced by different ResNet models.
* **class_specific_models.ipynb**: In this approach, we use image embeddings and K-means clustering to assign implicit class labels to each sample of the given dataset. We then train a separate model for each implicit class.
* **google_maps_processing.ipynb**: This notebook generates many new samples from the downloaded Google Maps data by slicing the large images into smaller ones and applying rotations and flips.
* **large_dataset_training.ipynb**: Here we use additional Google Maps data to train our models with much larger datasets. We first train the model on our new dataset and then finetune it with the original dataset. We also apply morphological operations for post-processing.
* **dilation.ipynb**: Experiment the use of dilation in the bottleneck part of the models. We use two similar approaches.
