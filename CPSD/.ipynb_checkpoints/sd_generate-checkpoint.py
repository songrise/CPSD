# -*- coding : utf-8 -*-
# @FileName  : sd_generate.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Mar 14, 2024
# @Github    : https://github.com/songrise
# @Description: sd generate a batch of images from a given prompt.

from models.sds import StableDiffusion
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
from torchvision.utils import save_image
import argparse
import PIL.Image as Image
from torchvision.utils import make_grid
import numpy


def sd_forward(sd, prompt):
    # prompt_ = "front view of the body of the Hulk wearing blue jeans, photorealistic style."

    # fix_randomness(42)
    latent = nn.Parameter(torch.empty(1, 4, 64, 64, device=device))

    img = sd.prompt_to_img(prompt, 512, 512, 100, return_torch=True, debug_dump=True)

    return img


if __name__ == "__main__":
    device = torch.device("cuda")
    sd = StableDiffusion(device, version="2.1")
    prompt_path = "/root/autodl-tmp/CPSD/data/prompt.txt"
    out_path = "data/sd21_generated_images"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    i = 0
    with open(prompt_path, "r") as f:
        # for each line in the prompt file
        for prompt in f:
            prompt = prompt.strip()
            print(f"Generating image for prompt: {prompt}")
            img = sd_forward(sd, prompt)

            save_image(img, f"{out_path}/img_{i}.png")
            print(f"Done, Image saved as {out_path}/img_{i}.png")
            i += 1
