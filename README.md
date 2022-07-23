# CIL Road Segmentation
## Kaggle Team: WeNeedANameHere

### Setup:
* TODO: Explain Python setup.
* Place the training and test folders of the dataset into the same directory as the Python notebooks.

### Notes:
Almost all of the main code is located in the *src* directory in order to keep the notebooks as clean, minimal and comprehensible as possible. We also deleted some output cells that contained many images to keep the file sizes of the notebooks small (e.g. output cells that show validation samples after each training epoch or all predictions on the test set).

### Notebooks:
* **unet-baseline-improvements.ipynb**: In this notebook, we take the U-Net baseline model from tutorial 10 as a starting point and build upon it by introducing and evaluating one improvement after the other.
* **augmentation-experiments.ipynb**: This notebook visualizes the dataset augmentation methods we use for the other notebooks, such as flips, rotations, reflection paddings and color jitters. The main purpose of this notebook is to experiment with different parameters.