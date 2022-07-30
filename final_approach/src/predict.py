import os
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import utils
from glob import glob
import cv2

import torch
from torch.utils.data import DataLoader

import utils
from configuration import Configuration
from configuration import CONSTANTS as C

import monai
from monai.inferers import sliding_window_inference
from monai.data import list_data_collate, decollate_batch
from monai.transforms import (
    Activations,
    AsDiscrete,
    AsChannelFirst,
    Compose,
    LoadImage,
    ScaleIntensity,
    EnsureType,
)

import segmentation_models_pytorch as smp

from mask_to_submission import get_submission
from absl import app, flags

start = time.time()

test_imgs = sorted(glob(os.path.join(C.CIL_DATA_DIR, "test", "images", "*.png")))
    
test_transforms = Compose(
    [LoadImage(image_only=True),
     AsChannelFirst(),
     ScaleIntensity(),
     EnsureType()]
)

post_transforms = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

test_ds = monai.data.Dataset(data=test_imgs, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1, collate_fn=list_data_collate)

"""Load the trained model"""
net = smp.Unet(encoder_name='resnet34',
               encoder_depth=5,
               encoder_weights='imagenet',
               decoder_use_batchnorm=True,
               decoder_channels=(512, 512, 256, 128, 64),
               in_channels=3,
               classes=1).to(C.DEVICE)
print('Model created with {} trainable parameters'.format(utils.count_parameters(net)))
pretrain_dir = utils.get_model_dir(C.EXPERIMENT_DIR, 1658858769)
pretrain_ckpt_file = os.path.join(pretrain_dir, 'model.pth')
    
pretrain_ckpt = torch.load(pretrain_ckpt_file)
net.load_state_dict(pretrain_ckpt['model_state_dict'])
print("Trained model loaded from {}".format(pretrain_ckpt_file))

print("Begin the Inference...")

final_output_lst = []
net.eval()
with torch.no_grad():
    for i, test_img in enumerate(test_loader):
        roi_size = (256, 256)
        sw_batch_size = 4
        
        output_lst = []
        
        for j in range(0, 4):
            test_img_rot = torch.rot90(test_img, k=j, dims = [2, 3])
            test_img_flip = torch.flip(test_img_rot, dims=[2,3])
            
            test_img_rot = test_img_rot.to(C.DEVICE)
            output = sliding_window_inference(test_img_rot, roi_size, sw_batch_size, net, overlap=0.75)
            output = [post_transforms(i) for i in decollate_batch(output)][0]
            output = np.swapaxes(output.cpu().detach().numpy(), 0, -1)*255
            output_lst.append(np.rot90(output, k=j+4))
            
            test_img_flip = test_img_flip.to(C.DEVICE)
            output = sliding_window_inference(test_img_flip, roi_size, sw_batch_size, net, overlap=0.75)
            output = [post_transforms(i) for i in decollate_batch(output)][0]
            output = np.swapaxes(output.cpu().detach().numpy(), 0, -1)*255
            output_lst.append(np.rot90(np.flip(output, axis=[0,1]), k=j+4))
        
        final_output = np.average(np.stack(output_lst, axis=0), axis=0)
        final_output = np.where(final_output > (255/2), 255, 0)
        final_output_lst.append(final_output)

for i, output in enumerate(final_output_lst):
    path = test_imgs[i].replace('images', 'prediction')
    cv2.imwrite(path, output)

pred_dir = os.path.join(C.CIL_DATA_DIR, "test", "prediction")

print("Predicted images are written to", pred_dir)

elapsed = time.time() - start
print('The Inference took: {:.2f} secs'.format(elapsed))

# Method to run mask_to_submission.py
get_submission()
print("Submission.csv written to the home directory.")
