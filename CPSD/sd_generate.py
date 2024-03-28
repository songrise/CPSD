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

    img = sd.prompt_to_img(prompt, 512, 512, 200, return_torch=True, debug_dump=False)

    return img


if __name__ == "__main__":
    from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
    import torch

    model_id = "stabilityai/stable-diffusion-2-1-base"

    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, scheduler=scheduler, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    prompt_path = "/root/autodl-tmp/CPSD/data/prompt.txt"
    out_path = "/root/autodl-tmp/CPSD/data/sd21_generated_images"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    i = 0
    with open(prompt_path, "r") as f:
        # for each line in the prompt file
        for prompt in f:
            prompt = prompt.strip()
            print(f"Generating image for prompt: {prompt}")
            img = pipe(prompt).images[0]
            img.save(f"{out_path}/img_{i}.png")
            print(f"Done, Image saved as {out_path}/img_{i}.png")
            i += 1
