# -*- coding : utf-8 -*-
# @FileName  : add_noise.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Jun 17, 2024
# @Github    : https://github.com/songrise
# @Description: add noise to image to visualize the effect of noise on image

#%%
import torchvision
import torch
import numpy as np
import PIL

src_image = "/root/autodl-tmp/CPSD/out/sd_style/main/96c8d0cd/out_6_2.png"
src_image = PIL.Image.open(src_image)
src_image = torchvision.transforms.ToTensor()(src_image)
H, W  = src_image.shape[1], src_image.shape[2]
noise = torch.randn(3, H, W) * 0.5
noisy_image = src_image + noise
noisy_image = torch.clamp(noisy_image, 0, 1)
noisy_image = torchvision.transforms.ToPILImage()(noisy_image)
noisy_image.show()
