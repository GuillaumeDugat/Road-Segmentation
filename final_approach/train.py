import os
import re
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import utils
from glob import glob

import torch
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

import utils
from configuration import Configuration
from configuration import CONSTANTS as C

import segmentation_models_pytorch as smp

import monai
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.data import list_data_collate, decollate_batch
from monai.visualize import plot_2d_or_3d_image
from monai.transforms import (
    Activations,
    AddChanneld,
    AsDiscrete,
    AsChannelFirstd,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    RandFlipd,
    Resized,
    ScaleIntensityd,
    EnsureTyped,
    EnsureType,
)


def main(config):
    # Fix seed.
    if config.seed is None:
        config.seed = int(time.time())

    # Create a new experiment ID and a folder where to store logs and config.
    experiment_id = int(time.time())
    model_dir = utils.create_model_dir(C.EXPERIMENT_DIR, experiment_id)

    ext_train_transforms = Compose(
        [LoadImaged(keys=["img", "seg"]),
         AddChanneld(keys=['seg']),
         AsChannelFirstd(keys=["img"]),
         ScaleIntensityd(keys=["img", "seg"]),
         # Resized(keys=["img", "seg"], spatial_size=[2500,2500], mode=['area', 'nearest']),
         # RandCropByPosNegLabeld(keys=["img", "seg"], label_key="seg", spatial_size=[96, 96],
         #                        pos=1, neg=0.4, num_samples=50),  # number of cropped samples
         # RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
         # RandFlipd(keys=["img", "seg"], prob=0.5, spatial_axis=None),
         EnsureTyped(keys=["img", "seg"])]
    )

    cil_train_transforms = Compose(
        [LoadImaged(keys=["img", "seg"]),
         AddChanneld(keys=['seg']),
         AsChannelFirstd(keys=["img"]),
         ScaleIntensityd(keys=["img", "seg"]),
         # Resized(keys=["img", "seg"], spatial_size=[2500,2500], mode=['area', 'nearest']),
         RandCropByPosNegLabeld(keys=["img", "seg"], label_key="seg", spatial_size=[256, 256],
                                pos=1, neg=1, num_samples=3),  # number of cropped samples
         # RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
         # RandFlipd(keys=["img", "seg"], prob=0.5, spatial_axis=None),
         EnsureTyped(keys=["img", "seg"])]
    )

    val_transforms = Compose(
        [LoadImaged(keys=["img", "seg"]),
         AddChanneld(keys=['seg']),
         AsChannelFirstd(keys=["img"]),
         ScaleIntensityd(keys=["img", "seg"]),
         EnsureTyped(keys=["img", "seg"])]
    )

    post_transforms = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    dg_train_imgs = sorted(glob(os.path.join(C.DEEPGLOBE_DATA_DIR, "train", "images_new", "*.png")))
    dg_train_segs = sorted(glob(os.path.join(C.DEEPGLOBE_DATA_DIR, "train", "groundtruth_new", "*.png")))
    assert len(dg_train_imgs) == len(dg_train_segs)

    mass_train_imgs = sorted(glob(os.path.join(C.MASS_DATA_DIR, "train", "images_new", "*.png")))
    mass_train_segs = sorted(glob(os.path.join(C.MASS_DATA_DIR, "train", "groundtruth_new", "*.png")))
    assert len(mass_train_imgs) == len(mass_train_segs)

    """Combine two external dataset"""
    ext_train_imgs = sorted(dg_train_imgs + mass_train_imgs)
    ext_train_segs = sorted(dg_train_segs + mass_train_segs)
    ext_train_files = [{"img": img, "seg": seg}
                       for img, seg in zip(ext_train_imgs, ext_train_segs)]

    print("DeepGlobe Length: ", len(dg_train_imgs), "; Mass Length: ", len(mass_train_imgs),
          "; Combined Length: ", len(ext_train_files))

    """CIL TRAIN/VALID SPLIT"""
    cil_train_imgs = glob(os.path.join(C.CIL_DATA_DIR, "train", "images", "satimage_*.png"))
    cil_train_segs = glob(os.path.join(C.CIL_DATA_DIR, "train", "groundtruth", "satimage_*.png"))
    cil_train_imgs.sort(key=lambda f: int(re.sub('\D', '', f)))
    cil_train_segs.sort(key=lambda f: int(re.sub('\D', '', f)))

    setattr(config, 'n_val', 352)
    cil_train_files = [{"img": img, "seg": seg}
                       for img, seg in zip(cil_train_imgs[:-config.n_val], cil_train_segs[:-config.n_val])]
    cil_val_files = [{"img": img, "seg": seg}
                     for img, seg in zip(cil_train_imgs[-config.n_val:], cil_train_segs[-config.n_val:])]
    print("CIL Length: ", len(cil_train_files), "+", len(cil_val_files))

    # Create data loader
    setattr(config, 'bs_ext_train', 42)
    ext_train_ds = monai.data.Dataset(data=ext_train_files, transform=ext_train_transforms)
    ext_train_loader = DataLoader(ext_train_ds,
                                  batch_size=config.bs_ext_train, shuffle=True,
                                  num_workers=config.data_workers,
                                  collate_fn=list_data_collate)
    print('External Train Dataloader len: ', len(ext_train_loader))

    setattr(config, 'bs_cil_train', 1)
    cil_train_ds = monai.data.Dataset(data=cil_train_files, transform=cil_train_transforms)
    cil_train_loader = DataLoader(cil_train_ds,
                                  batch_size=config.bs_cil_train, shuffle=True,
                                  num_workers=config.data_workers,
                                  collate_fn=list_data_collate)
    print('CIL Train Dataloader len: ', len(cil_train_loader))

    setattr(config, 'bs_eval', 1)
    cil_val_ds = monai.data.Dataset(data=cil_val_files, transform=val_transforms)
    cil_val_loader = DataLoader(cil_val_ds,
                                batch_size=config.bs_eval, shuffle=True,
                                num_workers=config.data_workers,
                                collate_fn=list_data_collate)
    print('CIL Valid Dataloader len: ', len(cil_val_loader))

    """CREATE UNET"""
    net = smp.Unet(encoder_name='resnet34',
                   encoder_depth=5,
                   encoder_weights='imagenet',
                   decoder_use_batchnorm=True,
                   decoder_channels=(512, 512, 256, 128, 64),
                   in_channels=3,
                   classes=1).to(C.DEVICE)
    print('Model created with {} trainable parameters'.format(utils.count_parameters(net)))

    pretrain_dir = utils.get_model_dir(C.EXPERIMENT_DIR, 1658825949)
    pretrain_ckpt_file = os.path.join(pretrain_dir, 'model.pth')
    if not os.path.exists(pretrain_ckpt_file):
        raise ValueError("Could not find model checkpoint {}.".format(pretrain_ckpt_file))
    pretrain_ckpt = torch.load(pretrain_ckpt_file)
    net.load_state_dict(pretrain_ckpt['model_state_dict'])
    print('Loaded pre-trained weights from {}'.format(pretrain_ckpt_file))

    """STORE EVERYTHING"""

    config.to_json(os.path.join(model_dir, 'config.json'))

    # Create a checkpoint file for the best model.
    checkpoint_file = os.path.join(model_dir, 'model.pth')
    print('Saving checkpoints to {}'.format(checkpoint_file))

    # Create Tensorboard logger.
    writer = SummaryWriter(os.path.join(model_dir, 'logs'))

    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    loss_function = monai.losses.DiceLoss(sigmoid=True)

    best_valid_metric = -float('inf')
    early_stop_count = 0

    for epoch in range(config.n_epochs):
        epoch_loss = 0
        epoch_len = len(ext_train_loader)
        start = time.time()
        net.train()

        cil_loader_iterator = iter(cil_train_loader)

        for i, ext_abatch in enumerate(ext_train_loader):
            optimizer.zero_grad()
            ext_inputs, ext_labels = ext_abatch["img"], ext_abatch["seg"]

            try:
                cil_abatch = next(cil_loader_iterator)
            except StopIteration:
                cil_loader_iterator = iter(cil_train_loader)
                cil_abatch = next(cil_loader_iterator)

            cil_inputs, cil_labels = cil_abatch["img"], cil_abatch["seg"]

            inputs = torch.cat((ext_inputs, cil_inputs), dim=0).to(C.DEVICE)
            labels = torch.cat((ext_labels, cil_labels), dim=0).to(C.DEVICE)

            if epoch == 0 and i == 0:
                print('ext shape:', ext_inputs.size()[0], 'cil shape:', cil_inputs.size()[0])
                print('inputs shape:', inputs.size(), 'labels shape:', labels.size())

            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            writer.add_scalar("Train/loss", loss.item(), epoch * epoch_len + i)

        elapsed = time.time() - start
        epoch_loss /= len(ext_train_loader)
        writer.add_scalar("Train/epoch_loss", epoch_loss, epoch)
        print('[Train {:0>3d}] loss: {:.4f} elapsed: {:.2f} secs'.format(
            epoch + 1, epoch_loss, elapsed))

        if (epoch + 1) % config.eval_every == 0:
            start = time.time()
            net.eval()
            with torch.no_grad():
                for val_data in cil_val_loader:
                    val_images, val_labels = val_data["img"].to(C.DEVICE), val_data["seg"].to(C.DEVICE)
                    roi_size = (256, 256)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, net, overlap=0.75)
                    val_outputs = [post_transforms(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

                valid_metric = dice_metric.aggregate(reduction="mean").item()
                elapsed = time.time() - start
                print('[Valid {:0>3d}] metric: {:.4f} elapsed: {:.2f} secs'.format(
                    epoch + 1, valid_metric, elapsed))
                dice_metric.reset()
                writer.add_scalar("Val/mean_dice", valid_metric, epoch + 1)

                if valid_metric > best_valid_metric:
                    best_valid_metric = valid_metric
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': epoch_loss,
                        'valid_metric': best_valid_metric,
                    }, checkpoint_file)
                    print('Saved checkpoints with metric {:.4f}'.format(best_valid_metric))
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                    print("Early stopping count: ", early_stop_count)

                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

            net.train()

        if early_stop_count > config.early_stop_threshold:
            print("EARLY STOPPED!")
            final_ckpt_file = os.path.join(model_dir, 'model_final.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss,
                'valid_metric': best_valid_metric,
            }, final_ckpt_file)
            print('Saved FINAL checkpoints with metric {:.4f}'.format(valid_metric))
            break


if __name__ == '__main__':
    main(Configuration.parse_cmd())
