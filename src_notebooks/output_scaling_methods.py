import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image

import torch
from torch import nn
import torchvision.transforms.functional as TF
import cv2

from src_notebooks.pytorch_utils import ImageDataset, patch_accuracy_fn, patch_f1_fn


def predict_by_resizing(model,
                        image_folder="training",
                        implicit_class_labels=None, 
                        resize_shape=(400, 400),
                        normalize=False):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_fns = sorted(glob(os.path.join(image_folder, "images", "*.png")))
    input_dataset = ImageDataset(train_fns, device=device, resize_to=resize_shape, normalize=normalize)
    input_dataloader = torch.utils.data.DataLoader(input_dataset, batch_size=1, shuffle=False, pin_memory=True)

    
    # whether or not to use multiple models based on implicit class lables
    use_specialized_models = isinstance(model, list) and implicit_class_labels is not None 


    sigmoid = nn.Sigmoid() # we need to manually apply the sigmoid function because it is not included in the models themselves,
                           # that's because the BCEWithLogitsLoss already applies the sigmoid function internally
    
    predictions = []

    model.eval()
    with torch.no_grad():
        for x in input_dataloader:

            if use_specialized_models:
                pred = model[implicit_class_labels[i]](x)
            else:
                pred = model(x)

            predictions.append(sigmoid(pred).detach().cpu().numpy())

    predictions = np.concatenate(predictions, 0)
    predictions = np.moveaxis(predictions, 1, -1)  # CHW to HWC
    predictions = np.stack([cv2.resize(img, dsize=(400, 400)) for img in predictions], 0)  # resize to original shape
    return predictions


def predict_by_parts(model,
                     image_folder="training",
                     implicit_class_labels=None,
                     input_size=384,
                     normalize=False):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_fns = sorted(glob(os.path.join(image_folder, "images", "*.png")))

    images = [ Image.open(x).convert("RGB") for x in train_fns] # size (400,400)

    use_specialized_models = isinstance(model, list) and implicit_class_labels is not None 
    sigmoid = nn.Sigmoid()
    
    predictions = []

    model.eval()
    with torch.no_grad():
        for i in range(len(images)):
            # split image into 4 correctly sized parts covering the entire image
            tl = TF.to_tensor(images[i].crop((0,0,input_size,input_size)))
            tr = TF.to_tensor(images[i].crop((400-input_size,0,400,input_size)))
            bl = TF.to_tensor(images[i].crop((0,400-input_size,input_size,400)))
            br = TF.to_tensor(images[i].crop((400-input_size,400-input_size,400,400)))
            batch_i = torch.stack([tl,tr,bl,br])
            batch_i = batch_i.to(device)

            if use_specialized_models:
                pred = model[implicit_class_labels[i]](batch_i)
            else:
                pred = model(batch_i)

            pred = sigmoid(pred)
            predictions.append(pred.detach().cpu().numpy())

    predictions = np.squeeze(np.stack(predictions, 0))
    
    return predictions

def stitch_predictions(part_predictions, seam_offset=16*3):
    stitched_predictions = []
    for parts in part_predictions:
        stitched = np.zeros((400,400))
        stitched[0:400-seam_offset, 0:400-seam_offset] = parts[0][0:400-seam_offset, 0:400-seam_offset]
        stitched[-seam_offset:400, 0:400-seam_offset] = parts[2][-seam_offset:, 0:400-seam_offset]
        stitched[0:400-seam_offset, -seam_offset:400] = parts[1][0:400-seam_offset, -seam_offset:]
        stitched[-seam_offset:400, -seam_offset:400] = parts[3][-seam_offset:, -seam_offset:]
        stitched_predictions.append(stitched)
    return np.stack(stitched_predictions,0)

def blend_predictions(part_predictions, blend_weight_mask=None):
    mask = blend_weight_mask
    if(blend_weight_mask == None):
        mask = np.zeros((400,400))
        mask[:384,:384] = 1.0
        mask[16:,:384] += 1.0
        mask[:384,16:] += 1.0
        mask[16:,16:] += 1.0
        mask = 1.0 / mask

    blended_predictions = []
    for parts in part_predictions:
        blended = np.zeros((400,400))
        blended[:384,:384] = mask[:384,:384] * parts[0]
        blended[16:,:384] += mask[16:,:384] * parts[2]
        blended[:384,16:] += mask[:384,16:] * parts[1]
        blended[16:,16:] += mask[16:,16:] * parts[3]
        blended_predictions.append(blended)
    return np.stack(blended_predictions,0)


def patch_accuracy_metrics(y,y_hat):
    worst_index = 0
    min_score = 101.0
    best_index = 0
    max_score = -1.0
    avg_score = 0
    for i in range(len(y)):
        score = patch_accuracy_fn(y_hat[i],y[i])
        avg_score += score
        if(score < min_score):
            min_score = score
            worst_index = i
        if(score > max_score):
            max_score = score
            best_index = i

    return (avg_score / len(y), min_score, max_score, worst_index, best_index)

def patch_f1_metrics(y,y_hat):
    worst_index = 0
    min_score = 101.0
    best_index = 0
    max_score = -1.0
    avg_score = 0
    for i in range(len(y)):
        score = patch_f1_fn(y_hat[i],y[i])
        avg_score += score
        if(score < min_score):
            min_score = score
            worst_index = i
        if(score > max_score):
            max_score = score
            best_index = i
    return (avg_score / len(y), min_score, max_score, worst_index, best_index)


def show_label_samples(y, y_resized, y_stitched, y_blended, indices=[0], segmentation=False, no_score=False):
    # no_score = True if y is not a groundtruth image. but instead a input image.
    assert(y.shape[-1] == 2 or no_score == True, "Can't compute score from input image as y. Please provide a valid groundtruth, or use no_score=true")
    
    fig, axs = plt.subplots(min(len(indices),len(y)),4, figsize=(18.5, 4.5*len(indices)),squeeze=False)
    
    for i,index in enumerate(indices):
        if(no_score==False):
            resize_score = patch_f1_fn(y_resized[index],y[index],cutoff=0.25)
            stitch_score = patch_f1_fn(y_stitched[index],y[index],cutoff=0.25)
            blend_score  = patch_f1_fn(y_blended[index],y[index],cutoff=0.25)
        
        axs[i, 0].imshow(y[index])
        axs[i, 1].imshow(y_resized[index])
        axs[i, 2].imshow(y_stitched[index])
        axs[i, 3].imshow(y_blended[index])
        axs[i, 0].set_title(f'GT patch_f1')
        if(no_score == False):
            axs[i, 1].set_title(f'resized {resize_score*100:.2f}')
            axs[i, 2].set_title(f'stitched {stitch_score*100:.2f}')
            axs[i, 3].set_title(f'blended {blend_score*100:.2f}')
        else:
            axs[i, 1].set_title(f'resized')
            axs[i, 2].set_title(f'stitched')
            axs[i, 3].set_title(f'blended')
        axs[i, 0].set_axis_off()
        axs[i, 1].set_axis_off()
        axs[i, 2].set_axis_off()
        axs[i, 3].set_axis_off()

    plt.show()