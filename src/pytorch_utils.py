import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn
from tqdm.notebook import tqdm
import torchvision.transforms.functional as TF

from src.utils import load_image, create_submission


PATCH_SIZE = 16  # pixels per side of square patches
CUTOFF = 0.25  # minimum average brightness for a mask patch to be classified as containing road


# Dataset class that deals with loading the data and making it available by index.
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_fns, mask_fns, device, resize_to=(400, 400), normalize=False):
        self.image_fns = image_fns
        self.mask_fns = mask_fns
        self.device = device
        self.resize_to = resize_to
        self.normalize = normalize
        self.n_samples = len(image_fns)

    def _normalize(self, img):
        tensor = TF.to_tensor(img) # tensor has shape (3, H, W) now!
        mean = tensor.mean([1,2])
        std = tensor.std([1,2])

        return TF.normalize(tensor, mean=mean, std=std)
    
    def _to_device(self, img):
        if self.device == 'cpu':
            return img.cpu()
        else:
            return img.contiguous().pin_memory().to(device=self.device, non_blocking=True)

    def __getitem__(self, item):
        x = load_image(self.image_fns[item])
        y = load_image(self.mask_fns[item])

        if self.resize_to != (400, 400):
            x = cv2.resize(x, dsize=self.resize_to)
            y = cv2.resize(y, dsize=self.resize_to)
        
        if self.normalize:
            x = self._normalize(x) # x is a tensor now with shape (3, H, W)
        else:
            x = TF.to_tensor(x) # this also transorms x to a tensor with shape (3, H, W)
        y = TF.to_tensor(y)

        return self._to_device(x), self._to_device(y)

    def __len__(self):
        return self.n_samples


def show_val_samples(x, y, y_hat, segmentation=False):
    # training callback to show predictions on validation set
    imgs_to_draw = min(5, len(x))
    if x.shape[-2:] == y.shape[-2:]:  # segmentation
        fig, axs = plt.subplots(3, imgs_to_draw, figsize=(18.5, 12))
        for i in range(imgs_to_draw):
            axs[0, i].imshow(np.moveaxis(x[i], 0, -1))
            axs[1, i].imshow(np.concatenate([np.moveaxis(y_hat[i], 0, -1)] * 3, -1))
            axs[2, i].imshow(np.concatenate([np.moveaxis(y[i], 0, -1)]*3, -1))
            axs[0, i].set_title(f'Sample {i}')
            axs[1, i].set_title(f'Predicted {i}')
            axs[2, i].set_title(f'True {i}')
            axs[0, i].set_axis_off()
            axs[1, i].set_axis_off()
            axs[2, i].set_axis_off()
    else:  # classification
        fig, axs = plt.subplots(1, imgs_to_draw, figsize=(18.5, 6))
        for i in range(imgs_to_draw):
            axs[i].imshow(np.moveaxis(x[i], 0, -1))
            axs[i].set_title(f'True: {np.round(y[i]).item()}; Predicted: {np.round(y_hat[i]).item()}')
            axs[i].set_axis_off()
    plt.show()


def accuracy_fn(y_hat, y):
    return (y_hat.round() == y.round()).float().mean()


def patch_accuracy_fn(y_hat, y):
    # computes accuracy weighted by patches (metric used on Kaggle for evaluation)
    h_patches = y.shape[-2] // PATCH_SIZE
    w_patches = y.shape[-1] // PATCH_SIZE
    patches_hat = y_hat.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean((-1, -3)) > CUTOFF
    patches = y.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean((-1, -3)) > CUTOFF
    return (patches == patches_hat).float().mean()


# Trains the given model with the given filenames for the satellite images and masks
def train_model(
        model,
        image_fns_train, mask_fns_train, 
        image_fns_val, mask_fns_val, 
        n_epochs=35,
        batch_size=8,
        resize_shape=(384, 384),
        pos_weight=1.0  # how much more to weight the positive labels in the masks
    ):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = ImageDataset(image_fns_train, mask_fns_train, device, resize_to=resize_shape)
    val_dataset = ImageDataset(image_fns_val, mask_fns_val, device, resize_to=resize_shape)

    print(f"Number of training samples:\t{len(train_dataset)}")
    print(f"Number of validation samples:\t{len(val_dataset)}")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    metric_fns = {'acc': accuracy_fn, 'patch_acc': patch_accuracy_fn}
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

    # training loop
    history = {}  # collects metrics at the end of each epoch

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        # initialize metric list
        metrics = {'loss': [], 'val_loss': []}
        for k, _ in metric_fns.items():
            metrics[k] = []
            metrics['val_'+k] = []

        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_epochs}')

        # training
        model.train()
        for (x, y) in pbar:
            optimizer.zero_grad()  # zero out gradients
            y_hat = model(x)  # forward pass
            loss = loss_fn(y_hat, y)
            loss.backward()  # backward pass
            optimizer.step()  # optimize weights

            # log partial metrics
            metrics['loss'].append(loss.item())
            for k, fn in metric_fns.items():
                metrics[k].append(fn(y_hat, y).item())
            pbar.set_postfix({k: sum(v)/len(v) for k, v in metrics.items() if len(v) > 0})

        # validation
        model.eval()
        with torch.no_grad():  # do not keep track of gradients
            for (x, y) in val_dataloader:
                y_hat = model(x)  # forward pass
                loss = loss_fn(y_hat, y)
                
                # log partial metrics
                metrics['val_loss'].append(loss.item())
                for k, fn in metric_fns.items():
                    metrics['val_'+k].append(fn(y_hat, y).item())

        # summarize metrics, log to tensorboard and display
        history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}

        print(' '.join(['\t- '+str(k)+' = '+str(v)+'\n ' for (k, v) in history[epoch].items()]))
        show_val_samples(x.detach().cpu().numpy(), y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())

    print('Finished Training')

    # plot loss curves
    plt.plot([v['loss'] for k, v in history.items()], label='Training Loss')
    plt.plot([v['val_loss'] for k, v in history.items()], label='Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


# Makes predictions on test set, creates submission file and returns the predicted masks.
# To predict with multiple models based on implicit class labels, pass a list of models for the 'model' argument.
def predict_on_test_set(model, implicit_class_labels=None, submission_fn='submission.csv'):
    def _np_to_tensor(x, device):
        if device == 'cpu':
            return torch.from_numpy(x).cpu()
        else:
            return torch.from_numpy(x).contiguous().pin_memory().to(device=device, non_blocking=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # predict on test set
    test_fns = sorted(glob(os.path.join("test", "images", "*.png")))
    test_images = np.stack([np.array(Image.open(fn)) for fn in test_fns]).astype(np.float32) / 255.0

    # we also need to resize the test images. This might not be the best ideas depending on their spatial resolution.
    test_images = np.stack([cv2.resize(img, dsize=(384, 384)) for img in test_images], 0)
    test_images = test_images[:, :, :, :3]
    test_images = _np_to_tensor(np.moveaxis(test_images, -1, 1), device)

    # whether or not to use multiple models based on implicit class lables
    use_specialized_models = isinstance(model, list) and implicit_class_labels is not None 

    test_pred = []
    for i in range(test_fns):
        if use_specialized_models:
            test_pred.append(model[implicit_class_labels[i]](test_images[i].unsqueeze(0)).detach().cpu().numpy())
        else:
            test_pred.append(model(test_images[i].unsqueeze(0)).detach().cpu().numpy())

    test_pred = np.concatenate(test_pred, 0)
    test_pred= np.moveaxis(test_pred, 1, -1)  # CHW to HWC
    test_pred = np.stack([cv2.resize(img, dsize=(400, 400)) for img in test_pred], 0)  # resize to original shape

    # now compute labels
    labels = test_pred.reshape((-1, 400 // PATCH_SIZE, PATCH_SIZE, 400 // PATCH_SIZE, PATCH_SIZE))
    labels = np.moveaxis(labels, 2, 3)
    labels = np.round(np.mean(labels, (-1, -2)) > CUTOFF)
    create_submission(labels, test_fns, submission_filename=submission_fn)

    print(f"Created submission file: {submission_fn}")

    return test_pred