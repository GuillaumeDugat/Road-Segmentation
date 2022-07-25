# CIL Road Segmentation
## Kaggle Team: WeNeedANameHere

### Setup:
* Since we only use the standard Python packages for ML/DL and no exotic packages or other installations, we simply assume that the reader already has a Python setup that is able to run our notebooks.
* Place the training and test folders of the dataset into the same directory as the Python notebooks.
* Go to the following link, download ***large.zip*** and place the unzipped folder into the same directory as the Python notebooks: https://drive.google.com/drive/folders/1GNzTsWcSkkLJfd1lcTLmzBma9KkU2oxH?usp=sharing

### Notes:
* Almost all of the main code is located in the *src* directory in order to keep the notebooks as clean, minimal and comprehensible as possible. 
* We deleted some output cells that contained many images to keep the file sizes of the notebooks small (e.g. output cells that show validation samples after each training epoch or all predictions on the test set).
* For completeness, we added the code for the retrieval of additional data using the Google Maps API (*src/google_maps_download.py*). However, to protect our API key and money, we ask the reader to download the pre-downloaded images from the Google Drive link above.
* If image normalization is applied as a preprocessing step, the validation images shown after each epoch will look weird due to clipping since the normalized values are outside the valid range of pixel values.

### Notebooks:
* **augmentation_experiments.ipynb**: This notebook visualizes the dataset augmentation methods we use in other notebooks, such as flips, rotations, reflection paddings and color jitters. The main purpose of this notebook is to experiment with different parameters.
* **unet_baseline.ipynb**: In this notebook, we take the U-Net baseline model from tutorial 10 as a starting point and build upon it by introducing and evaluating one improvement after the other.
* **resnet_unet.ipynb**: We conduct experiments on a U-Net model where the down-sampling encoder is replaced by different ResNet models.
* **class_specific_models.ipynb**: In this approach, we use image embeddings and K-means clustering to assign implicit class labels to each sample of the given dataset. We then train a separate model for each implicit class.
* **google_maps_processing.ipynb**: This notebook generates many new samples from the downloaded Google Maps data by slicing the large images into smaller ones and applying rotations and flips.
* **large_dataset_training.ipynb**: Here we use the additional Google Maps data to train our models with much larger datasets.