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
    end_step=30,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=device,
    style_step=0,
    early_return_step=None,
    use_generative_style=False,
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

    # Set num inference steps
    pipe.scheduler.set_timesteps(50, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    latents = latents.repeat(len(prompt), 1, 1, 1)
    if use_generative_style:
        generative_latent = torch.randn(1, 4, 64, 64, device=device)
        generative_latent *= pipe.scheduler.init_noise_sigma
        latents[1] = generative_latent
    for i in tqdm(range(start_step, end_step)):

        t = pipe.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        noise_pred = pipe.unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        if i == early_return_step:
            return latents
    # Post-processing
    images = pipe.decode_latents(latents)
    images = pipe.numpy_to_pil(images)

    return images


if __name__ == "__main__":
    # seed all

    content_prompt = [
        "a cute fluffy cat with bright green eyes playing",
        "a tall glass skyscraper reaching high into the clouds",
        "crowded street with diverse people walking and shopping",
        "experienced lawyer with gray hair carrying a worn briefcase",
        "confident young woman in red dress walking city sidewalk",
        "delicious pizza with melted cheese and fresh colorful toppings",
        "serene mountain landscape with snowy peaks and clear lake",
        "colorful flower garden with roses, tulips, and exotic blooms",
        "busy city intersection with cars, pedestrians, and traffic lights",
        "cozy wooden cabin in forest clearing with smoking chimney",
        # "majestic lion surveying territory from rock in African savanna",
        # "peaceful beach with white sand, blue waves, and palms",
        # "futuristic cityscape with holographic billboards and flying vehicles",
        # "restored 1950s vintage automobile with chrome accents parked",
        # "dense rainforest with towering trees, colorful birds, and waterfall",
        # "charming small town with quaint shops and historic square",
        # "vibrant farmers market with produce stands and artisanal crafts",
        # "mysterious narrow alley between old buildings with flickering lamp",
        # "grand medieval castle with towers, drawbridge, and stone walls",
        # "tranquil Japanese garden with koi pond and stone lanterns",
        # "group of friends enjoying picnic on checkered blanket",
        # "daring skydiver in freefall with colorful parachute opening",
        # "cozy bookshop interior with shelves, armchairs, and reading nooks",
        # "stunning underwater coral reef teeming with colorful tropical fish",
        # "bustling open-air market with vendors, produce, and handmade crafts",
        # "serene lakeside cabin surrounded by pine trees and mountains",
        # "historic European town square with fountain and old buildings",
        # "vibrant street art mural covering entire side of building",
        # "majestic snow-capped mountain peak glowing orange at sunrise",
        # "cozy coffee shop with steamy windows on rainy day",
        # "lively music festival scene with crowd and colorful stage",
        # "tranquil bamboo forest with sunlight filtering through leaves",
        # "colorful hot air balloons rising into clear blue sky",
        # "rustic red barn in countryside with rolling hills background",
        # "lively beach volleyball game with players diving for ball",
        # "serene Buddhist temple with golden statues and incense smoke",
        # "bustling city street at night with neon signs",
        # "cozy fireplace in log cabin with crackling flames",
        # "cute girl hugging fluffy pet dog in park",
        # "running cat chasing butterfly on sunny grass field",
        # "beautiful sunset over ocean with silhouetted palm trees",
        # "cozy living room with fireplace and comfortable armchairs",
        # "delicious pizza with melted cheese and crispy pepperoni slices",
        # "serene mountain landscape bathed in golden sunset light",
        # "colorful flower garden in full bloom with butterflies",
        # "busy city intersection at rush hour with cars",
        # "steaming cup of coffee on rustic wooden table",
        # "majestic lion with full mane resting in savanna",
        # "peaceful beach scene at sunrise with gentle waves",
        # "futuristic cityscape with flying cars and towering skyscrapers",
    ]

    style_prompt = [
        "A watercolor style painting of",
        "A fauvism style painting of",
        "A B&W pencil sketch style painting of",
        "A pointillism style painting of",
        "An art deco style painting of",
        "A impressionism style painting of",
        "A vibrant ink-wash style painting of",
        "A pop-art style painting of",
        "A cubism style painting of",
        "An expressionism style painting of",
    ]
    # sample a image
    content_injection_layers = [
        None,
        # [0],
        # [0, 1, 2, 3],
        # [0, 1, 2, 3, 4, 5, 6, 7],
        # [0, 1, 2, 3, 4, 5],
    ]
    style_injection_layers = [
        None,
        # [0],
        [0, 1, 2, 3],
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        # [0, 1, 2, 3, 4, 5],
    ]
    # content_injection_layers = [[0, 1, 2, 3]]
    # style_injection_layers = [[0, 1, 2, 3, 4, 5, 6, 7]]
    total_step_T = 50
    for start_step_tau in [50, 45, 40, 35, 30, 25, 20, 15, 10, 5]:
        for c_i, c_layer in enumerate(content_injection_layers):
            # for s_i, s_layer in enumerate(style_injection_layers):
            s_layer = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            s_i = 1
            if s_layer == None:
                s_i = -1
            if c_layer == None:
                c_i = -1
            # c_layer = [0, 1, 2, 3]
            # c_i = 1
            exp_utils.seed_all(42)
            for i, cp in enumerate(content_prompt):
                # if i < 5:
                # continue
                for j, sp in enumerate(style_prompt):
                    x_T = torch.randn(1, 4, 64, 64, device="cuda")
                    x_T *= pipe.scheduler.init_noise_sigma
                    x_tau = sample(
                        prompt=[cp],
                        start_step=0,
                        start_latents=x_T,
                        guidance_scale=7.5,
                        end_step=total_step_T,  # will early return
                        do_classifier_free_guidance=True,
                        negative_prompt="",
                        device=device,
                        style_step=0,
                        early_return_step=total_step_T - start_step_tau,
                    )

                    attn_injection.register_attention_processors(
                        pipe,
                        share_attn=True if s_layer is not None else False,
                        share_cross_attn=True if c_layer is not None else False,
                        share_resnet_layers=c_layer,
                        share_attn_layers=s_layer,
                        share_value=False,
                        use_adain=True,
                    )

                    # full_style_prompt = sp + " " + cp
                    prompt_in = [cp, sp, ""]
                    x_0s = sample(
                        start_step=total_step_T - start_step_tau + 1,
                        start_latents=x_tau,
                        guidance_scale=7.5,
                        end_step=50,
                        do_classifier_free_guidance=True,
                        negative_prompt="",
                        device=device,
                        style_step=0,
                        prompt=prompt_in,
                        use_generative_style=True,
                    )

                    content, style, final = x_0s  # PIL images

                    out_folder = f"/root/autodl-tmp/CPSD/vis_out/ablate_delegation/tau_{start_step_tau}/content_{c_i}/"
                    os.makedirs(out_folder, exist_ok=True)
                    content.save(
                        f"{out_folder}/{i}_{j}_content.png",
                    )
                    final.save(
                        f"{out_folder}/{i}_{j}_final.png",
                    )

                    print(f"Saved to {out_folder}")
                    attn_injection.unset_attention_processors(pipe, True, True)
