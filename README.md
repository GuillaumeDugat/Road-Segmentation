# CIL Road Segmentation
## Kaggle Team: WeNeedANameHere

### Setup:
* Since we only use the standard Python packages for ML/DL and no exotic packages or other installations, we assume that the reader already has a Python setup that is able to run our notebooks.
* Simply place the training and test folders of the dataset into the same directory as the Python notebooks.

### Notes:
* Almost all of the main code is located in the *src* directory in order to keep the notebooks as clean, minimal and comprehensible as possible. 
* We deleted some output cells that contained many images to keep the file sizes of the notebooks small (e.g. output cells that show validation samples after each training epoch or all predictions on the test set).
* If image normalization is applied as a preprocessing step, the validation images shown after each epoch will look weird due to clipping since the normalized values are outside the valid range of pixel values.

### Notebooks:
* **augmentation_experiments.ipynb**: This notebook visualizes the dataset augmentation methods we use in other notebooks, such as flips, rotations, reflection paddings and color jitters. The main purpose of this notebook is to experiment with different parameters.
* **unet_baseline.ipynb**: In this notebook, we take the U-Net baseline model from tutorial 10 as a starting point and build upon it by introducing and evaluating one improvement after the other.
* **resnet_unet.ipynb**: We conduct experiments on a U-Net model where the down-sampling encoder is replaced by different ResNet models.
* **class_specific_models.ipynb**: In this approach, we use image embeddings and K-means clustering to assign implicit class labels to each sample of the given dataset. We then train a separate model for each implicit class.