import os
import sys
import random
import numpy as np

import torchvision.transforms as T
import torchvision.transforms.functional as TF


# fix randomness
random.seed(0)
np.random.seed(0)


def augment_dataset(
    imgs, 
    masks,
    n_samples_per_img=10, # how many new samples to generate from each original sample
    class_label=None, # optional implicit class label 
    dir="training_ext", # directory name of the new extended dataset
    angle=15, # after rotating by 0/90/180/270°, we rotate by another random degree from the range [-angle, angle]
    brightness=0.0,
    contrast=0.0,
    saturation=0.0,
    hue=0.0,
    ):
    
    # create directories for extended dataset
    os.makedirs(dir, exist_ok=True)
    os.makedirs(os.path.join(dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(dir, "groundtruth"), exist_ok=True)

    # instantiate color jitter with given parameters
    jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    # generate new samples
    counter = 0
    num_images = len(imgs) * n_samples_per_img
    for (img, mask) in zip(imgs, masks):
        for i in range(n_samples_per_img):
            img_size = img.size

            # reflection padding
            curr_img = TF.pad(img=img, 
                              padding=[img_size[0] // 5, img_size[1] // 5], 
                              padding_mode="reflect")
            curr_mask = TF.pad(img=mask, 
                              padding=[img_size[0] // 5, img_size[1] // 5], 
                              padding_mode="reflect")

            # rotation
            angle1 = random.choice([0, 90, 180, 270])
            angle2 = random.randint(-angle, angle)
            curr_img = TF.rotate(TF.rotate(curr_img, angle1), angle2)
            curr_mask = TF.rotate(TF.rotate(curr_mask, angle1), angle2)

            # flip
            if random.random() > 0.5:
                curr_img = TF.hflip(curr_img)
                curr_mask = TF.hflip(curr_mask)

            # center crop
            curr_img = TF.center_crop(curr_img, img_size)
            curr_mask = TF.center_crop(curr_mask, img_size)

            # color jitters
            curr_img = jitter(curr_img)

            # save image and mask
            counter_str = str(counter).zfill(5)
            fn = f"{counter_str}.png" if class_label is None else f"{class_label}_{counter_str}.png"
            curr_img.save(os.path.join(dir, "images", fn))
            curr_mask.save(os.path.join(dir, "groundtruth", fn))

            counter += 1
            sys.stdout.write(f"\rImage {counter}/{num_images}")
    print("")