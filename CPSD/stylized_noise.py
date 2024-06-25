# %%
import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, DDIMScheduler
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
from diffusers.schedulers import DDIMScheduler
import torch.nn.functional as F
from models import attn_injection
from omegaconf import OmegaConf
from typing import List, Tuple
import uuid
import omegaconf
import utils.exp_utils as exp_utils

device = torch.device("cuda")


# Useful function for later
def load_image(url, size=None):
    response = requests.get(url, timeout=0.2)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    if size is not None:
        img = img.resize(size)
    return img


# Load a pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base"
).to(device)


# Set up a DDIM scheduler
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


# Sample function (regular DDIM)
@torch.no_grad()
def sample(
    prompt,
    start_step=0,
    start_latents=None,
    guidance_scale=3.5,
    num_inference_steps=30,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=device,
    style_step=0,
    style_prompt=None,
):
    negative_prompt = [""] * len(prompt)
    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
    )

    style_text_embeddings = pipe._encode_prompt(
        style_prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
    )

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    latents = latents.repeat(len(prompt), 1, 1, 1)
    for i in tqdm(range(start_step, num_inference_steps)):

        t = pipe.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        used_text_embeddings = (
            text_embeddings if i < style_step else style_text_embeddings
        )
        if style_step == 0:
            used_text_embeddings = style_text_embeddings

        noise_pred = pipe.unet(
            latent_model_input, t, encoder_hidden_states=used_text_embeddings
        ).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        # Normally we'd rely on the scheduler to handle the supdate step:
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
    # Post-processing
    images = pipe.decode_latents(latents)
    images = pipe.numpy_to_pil(images)

    return images


if __name__ == "__main__":
    # seed all

    content_prompt = [
        "a cute cat",
        "a tall building",
        "many people on the street",
        "an old lawyer",
        "a young lady in a red dress",
        "a delicious pizza",
        "a serene mountain landscape",
        "a colorful flower garden",
        "a busy city intersection",
        "a rustic wooden cabin",
        "a majestic lion in the wild",
        "a peaceful beach scene",
        "a futuristic cityscape",
        "a vintage automobile",
        "a lush rainforest",
        "a charming small town",
        "a vibrant farmers market",
        "a mysterious dark alley",
        "a grand medieval castle",
        "a tranquil Japanese garden",
        "a group of friends enjoying a picnic",
        "a daring skydiver in action",
        "a cozy bookshop interior",
        "a stunning underwater coral reef",
        "a bustling open-air market",
        "a serene lakeside cabin",
        "a historic European town square",
        "a vibrant street art mural",
        "a majestic mountain peak at sunrise",
        "a cozy coffee shop on a rainy day",
        "a lively music festival scene",
        "a tranquil bamboo forest",
        "a colorful hot air balloon festival",
        "a rustic barn in a countryside setting",
        "a lively beach volleyball game",
        "a serene Buddhist temple",
        "a bustling city street at night",
        "a cozy fireplace in a log cabin",
        "a cute girl with a pet dog",
        "a running cat on a grass field",
        "a beautiful sunset over the ocean",
        "a cozy living room with a fireplace",
        "a delicious pizza with pepperoni",
        "a serene mountain landscape at sunset",
        "a colorful flower garden in full bloom",
        "a busy city intersection at rush hour",
        "a cup of coffee on a wooden table",
        "a majestic lion in the wild",
        "a peaceful beach scene at sunrise",
        "a futuristic cityscape with flying cars",
    ]

    style_prompt = [
        "in watercolor style",
        "in fauvism style",
        "in pencil sketch style",
        "in pointillism style",
        "in art deco style",
        "in impressionism style",
        "in surrealism style",
        "in pop art style",
        "in cubism style",
        "in abstract expressionism style",
    ]
    # sample a image
    global_step = 0
    for i, cp in enumerate(content_prompt):
        for j, sp in enumerate(style_prompt):
            for step in [50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0]:
                exp_utils.seed_all(40)
                prompt = cp
                prompt_2 = cp + " " + sp
                imgs = sample(
                    [prompt],
                    num_inference_steps=50,
                    style_prompt=[prompt_2],
                    style_step=step,
                    guidance_scale=7.5,
                )
                os.makedirs(
                    f"/root/autodl-tmp/CPSD/out/ablation/stylized_noise_2/{global_step}",
                    exist_ok=True,
                )
                for i, img in enumerate(imgs):
                    img.save(
                        f"/root/autodl-tmp/CPSD/out/ablation/stylized_noise_2/{global_step}/stylenoise_{50-step}.png"
                    )
                    print(
                        f"Image saved as /root/autodl-tmp/CPSD/out/ablation/stylized_noise_2/{global_step}/stylenoise_{50-step}.png"
                    )
                # also save the prompt as txt
                with open(
                    f"/root/autodl-tmp/CPSD/out/ablation/stylized_noise_2/{global_step}/prompt.txt",
                    "w",
                ) as f:
                    f.write(prompt)
                    f.write("\n")
                    f.write(prompt_2)

            global_step += 1
