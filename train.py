import argparse
import logging
import os
import pprint
import random
from tqdm import tqdm
import argparse

import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from torch.optim import Adam
import torch.nn.functional as F

from models.pix2pixhd  import Generator
from models.discriminator import Discriminator
from dataloader import Sat2RrdDataset 
from loss import GANLoss, SiLogLoss
from metrics import cal_csi
from utils import save_images

parser = argparse.ArgumentParser(description='PyTorch sat2rdr Training')
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
parser.add_argument('--batch_size', default=4, type=float, help='batch size')
parser.add_argument('--num_workers', default=4, type=float, help='cpu number')
parser.add_argument('--epoch', default=200, type=float, help='epoch')

# python3 train.py --random_crop True --random_hflip True --random_vflip True
parser.add_argument('--dataroot', default='/workspace/SSD_4T_c/Georain/MS/StegoGAN/dataset/', type=str, help='data path')
parser.add_argument('--random_crop', default=False, type=bool, help='Enable random cropping of images.')
parser.add_argument('--random_hflip', default=False, type=bool, help='Enable random horizontal flipping of images.')
parser.add_argument('--random_vflip', default=False, type=bool, help='Enable random vertical flipping of images.')
parser.add_argument('--channels', default=[0, 1, 2, 3], type=int, nargs='+', help='List of channel indices to use.')

args = parser.parse_args()


cudnn.benchmark = True

train_dataset = Sat2RrdDataset(dataroot=args.dataroot, phase='train', random_crop=args.random_crop, random_hflip=args.random_hflip, random_vflip=args.random_vflip, channels=args.channels)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

val_dataset = Sat2RrdDataset(dataroot=args.dataroot, phase='test', channels=args.channels)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

model_G = Generator(input_ch=4)
model_D = Discriminator(input_ch=4, output_ch=1)

model_G.to('cuda:0')
model_D.to('cuda:0')

criterion_gan = GANLoss()
criterion_siloss = SiLogLoss()

model_G_optim = torch.optim.Adam(model_G.parameters(), lr=args.lr, betas=(0.5, 0.999))
model_D_optim = torch.optim.Adam(model_D.parameters(), lr=args.lr, betas=(0.5, 0.999))

train_total_iters = args.epoch * len(train_dataloader) # n_epochs * len(data_loader)
val_total_iters = len(val_dataloader)

def train(epoch):
    epoch_loss_G = 0
    epoch_loss_D = 0

    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.epoch}", leave=False)

    for idx, batchs in pbar:
        model_G.train()
        model_D.train()

        inputs, targets = batchs['A'].to('cuda:0'), batchs['B'].to('cuda:0')
        pred = model_G(inputs)

        loss_D, loss_G, target, fake = criterion_gan(model_D, model_G, inputs, targets) # D, G, input, target
        loss_si = criterion_siloss(pred, targets)

        loss_G = loss_G + loss_si

        model_G_optim.zero_grad()
        loss_G.backward()
        model_G_optim.step()

        model_D_optim.zero_grad()
        loss_D.backward()
        model_D_optim.step()

        iters = epoch * len(train_dataloader) + idx
            
        lr = args.lr * (1 - iters / train_total_iters) ** 0.9

        model_G_optim.param_groups[0]["lr"] = lr
        model_D_optim.param_groups[0]["lr"] = lr

        epoch_loss_G += loss_G.item()
        epoch_loss_D += loss_D.item()

        pbar.set_postfix({"Loss_G": f"{loss_G.item():.4f}", "Loss_D": f"{loss_D.item():.4f}", "LR": f"{lr:.14f}"})

    torch.save(model_G.state_dict(), f'./results/{epoch}.pth')

def val(epoch):

    csi_1mm, pod_1mm, far_1mm = 0, 0, 0
    csi_4mm, pod_4mm, far_4mm = 0, 0, 0
    csi_8mm, pod_8mm, far_8mm = 0, 0, 0

    pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f"Epoch {epoch+1}/{args.epoch}", leave=False)

    total_squared_error = 0
    total_samples = 0

    for idx, batchs in pbar:
        model_G.eval()

        with torch.no_grad():
            inputs, targets = batchs['A'].to('cuda:0'), batchs['B'].to('cuda:0')
            A_path, B_path =  batchs['A_paths'], batchs['B_paths']
            pred = model_G(inputs)
            csi, pod, far = cal_csi(pred, targets, threshold=1.0)
            csi_1mm += csi
            pod_1mm += pod
            far_1mm += far

            csi, pod, far = cal_csi(pred, targets, threshold=4.0)
            csi_4mm += csi
            pod_4mm += pod
            far_4mm += far

            csi, pod, far = cal_csi(pred, targets, threshold=8.0)
            csi_8mm += csi
            pod_8mm += pod
            far_8mm += far

            total_squared_error += torch.sum((pred - targets) ** 2).item()
            total_samples += targets.numel()

        pbar.set_postfix({
            "CSI 1mm": f"{csi_1mm / (idx + 1):.4f}", 
            "CSI 4mm": f"{csi_4mm / (idx + 1):.4f}",
            "CSI 8mm": f"{csi_8mm / (idx + 1):.4f}"
        })
    # epoch, input_image, true_image, pred_image, A_path, B_path
    save_images(epoch, inputs[-1, 0, :,:], targets[-1,0,:,:], pred[-1,0,:,:], A_path[-1], B_path[-1])
    n_batches = len(val_dataloader)

    csi_1mm /= n_batches
    pod_1mm /= n_batches
    far_1mm /= n_batches

    csi_4mm /= n_batches
    pod_4mm /= n_batches
    far_4mm /= n_batches

    csi_8mm /= n_batches
    pod_8mm /= n_batches
    far_8mm /= n_batches

    rmse = (total_squared_error / total_samples) ** 0.5

    print(f"Validation Results Epoch [{epoch+1}/200]:")
    print(f"1mm - CSI: {csi_1mm:.4f}, POD: {pod_1mm:.4f}, FAR: {far_1mm:.4f}")
    print(f"4mm - CSI: {csi_4mm:.4f}, POD: {pod_4mm:.4f}, FAR: {far_4mm:.4f}")
    print(f"8mm - CSI: {csi_8mm:.4f}, POD: {pod_8mm:.4f}, FAR: {far_8mm:.4f}")
    print(f"RMSE: {rmse:.4f}")

if __name__ == "__main__":
    for epoch in range(args.epoch):
        train(epoch)
        val(epoch)

