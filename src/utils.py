import os
import re
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image


# Returns a specific image from the given path as a numpy array.
# Satellite images have shape (H, W, 3) and masks have shape (H, W).
# All pixel values are in the interval [0.0, 1.0].
def load_image(path):
    img = np.array(Image.open(path)).astype(np.float32) / 255.0

    # if img is an image, omit alpha channel
    if len(img.shape) == 3:
        return img[:,:,:3] # shape (H, W, 3)
    else:
        return img # shape (H, W)


# Returns all images from the given path in a numpy array.
def load_images(path):
    return np.array([load_image(fn) for fn in sorted(glob(path + '/*.png'))])


# Returns all images from the given path as PIL images.
def load_pil_images(path):
    return [Image.open(fn) for fn in sorted(glob(path + '/*.png'))]


# Plots all images from the given array of images.
def plot_images(images):
    num_images = len(images)
    
    fig, axs = plt.subplots(1, num_images, figsize=(4 * num_images, 5))
    for i in range(num_images):
        axs[i].imshow(images[i])
        axs[i].set_axis_off()
    plt.show()


# Plots all test images and the corresponding prediction masks.
def plot_predictions(preds, images_per_row=6):
    test_path = os.path.join("test", "images")
    test_images = load_images(test_path)

    n_full_rows = len(test_images) // images_per_row
    for i in range(n_full_rows):
        plot_images(test_images[i * images_per_row : (i + 1) * images_per_row])
        plot_images(preds[i * images_per_row : (i + 1) * images_per_row])

    # plot leftover
    if len(test_images) % images_per_row > 0:
        plot_images(test_images[i * n_full_rows:])
        plot_images(preds[i * n_full_rows:])


# Creates submission file with the given labels and test image filenames.
def create_submission(labels, test_filenames, submission_filename):
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn, patch_array in zip(test_filenames, labels):
            img_number = int(re.search(r"\d+", fn).group(0))
            for i in range(patch_array.shape[0]):
                for j in range(patch_array.shape[1]):
                    f.write("{:03d}_{}_{},{}\n".format(img_number, j * 16, i * 16, int(patch_array[i, j])))