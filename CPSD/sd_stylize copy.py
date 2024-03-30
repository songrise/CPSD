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
from diffusers.schedulers import DDIMScheduler
import torch.nn.functional as F
from models import attn_injection
import tqdm
from omegaconf import OmegaConf
from typing import List, Tuple


def sd_forward(sd, prompt):
    # prompt_ = "front view of the body of the Hulk wearing blue jeans, photorealistic style."

    # fix_randomness(42)
    latent = nn.Parameter(torch.empty(1, 4, 64, 64, device=device))

    img = sd.prompt_to_img(prompt, 512, 512, 100, return_torch=True, debug_dump=True)

    return img


def style_aligned_next_step(
    scheduler, noise_pred, timestep: int, latent, num_inference_steps
):
    timestep, next_timestep = (
        min(
            timestep - scheduler.config.num_train_timesteps // num_inference_steps,
            999,
        ),
        timestep,
    )
    alpha_prod_t = (
        scheduler.alphas_cumprod[int(timestep)]
        if timestep >= 0
        else scheduler.final_alpha_cumprod
    )
    alpha_prod_t_next = scheduler.alphas_cumprod[int(next_timestep)]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (latent - beta_prod_t**0.5 * noise_pred) / alpha_prod_t**0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * noise_pred
    next_sample = alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction
    return next_sample


if __name__ == "__main__":
    device = torch.device("cuda")
    sd = StableDiffusion(device, version="2.1")
    attn_injection.register_attention_processors(
        sd.unet,
        "pnp",
        "/root/autodl-tmp/CPSD/attn/debug",
        share_resblock=True,
        share_attn=True,
    )
    tau_attn = 10
    tau_feat = 10
    g_cpu = torch.Generator(device="cpu")
    g_cpu.manual_seed(10)

    out_path = "/root/autodl-tmp/CPSD/out/sd_style"
    src_img = "/root/autodl-tmp/plug-and-play/data/horse.png"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # load the source image as tensor
    src_img = Image.open(src_img)
    src_img = transforms.ToTensor()(src_img).unsqueeze(0).to(device)
    T = 50

    prompt = [
        "A white horse on a green field, photorealistic style.",
        "A white horse on a green field, sketch style.",
    ]

    text_embeddings = sd.get_text_embeds(prompt)

    ddim_sampler = DDIMScheduler(num_train_timesteps=1000)
    ddim_sampler.set_timesteps(T)
    h, w = src_img.shape[-2:]
    # zero pad to 512x512
    src_img_512 = torchvision.transforms.functional.pad(
        src_img, ((512 - w) // 2,), fill=0, padding_mode="constant"
    )
    src_img_512 = F.interpolate(src_img, (512, 512), mode="bilinear", align_corners=False)
    # drop the alpha channel if it exists
    if src_img_512.shape[1] == 4:
        src_img_512 = src_img_512[:, :3]
    z0 = sd.encode_imgs(src_img_512)
    # backward, add noise to the source image
    noise = torch.randn_like(z0)
    zts = [z0]
    # first ddim inversion
    latent = z0.clone().detach()
    for i in tqdm.trange(ddim_sampler.num_inference_steps):
        t_i = ddim_sampler.timesteps[len(ddim_sampler.timesteps) - i - 1]
        t_i = torch.tensor(t_i, device=src_img_512.device, dtype=torch.long)
        latent_model_input = torch.cat([latent] * 2)

        with torch.no_grad():
            noise_pred = sd.unet(
                latent_model_input, t_i, encoder_hidden_states=text_embeddings[:2, ...]
            )["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 2 * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            # latent = ddim_sampler.step(noise_pred, t_i, latent)["prev_sample"]
            latent = style_aligned_next_step(ddim_sampler, noise_pred, t_i, latent, T)
            # zt = ddim_sampler.add_noise(z0, noise, t_i)

            zts.append(latent)
    zts = torch.cat(zts).flip(0)

    # if True:  # decode and save the image at each ddim step
    #     all_imgs = []
    #     for i, latent in enumerate(zts):
    #         img = sd.decode_latents(latent.unsqueeze(0))
    #         all_imgs.append(img)
    #     # grid
    #     grid = make_grid(torch.cat(all_imgs), nrow=8)
    #     # grid = torchvision.transforms.functional.resize(grid)
    #     save_image(grid, f"{out_path}/ddim_grid.png")

    offset = 0
    zt = zts[offset].unsqueeze(0)
    latents = torch.cat([zt] * 2)
    # latents = torch.randn(
    #     2,
    #     4,
    #     64,
    #     64,
    # ).to("cuda:0")

    with torch.cuda.amp.autocast():
        for i in tqdm.trange(ddim_sampler.num_inference_steps - offset):
            # should not reverse, in descending order, 980, 960, ...,
            t_i = ddim_sampler.timesteps[i + offset]
            t_i = torch.tensor(t_i, device=src_img_512.device, dtype=torch.long)
            if (T - i) == tau_attn:
                attn_injection.unset_attention_processors(sd.unet, unset_share_attn=True)
            if (T - i) == tau_feat:
                attn_injection.unset_attention_processors(
                    sd.unet, unset_share_resblock=True
                )

            latents[0] = zts[i]
            # latents = torch.cat(
            #     [zts[i, ...].unsqueeze(0), latents[1, ...].unsqueeze(0)], dim=0
            # )  # the ddim inversed latent + style latent from the previous step
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.repeat_interleave(latents, 2, dim=0)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = sd.unet(
                    latent_model_input, t_i, encoder_hidden_states=text_embeddings
                )["sample"]

            # perform guidance
            (
                noise_pred_uncond_ref,
                noise_pred_text_ref,
                noise_pred_uncond_tgt,
                noise_pred_text_tgt,
            ) = noise_pred.chunk(4)
            noise_pred_ref = noise_pred_uncond_ref + 7.5 * (
                noise_pred_text_ref - noise_pred_uncond_ref
            )
            noise_pred_tgt = noise_pred_uncond_tgt + 7.5 * (
                noise_pred_text_tgt - noise_pred_uncond_tgt
            )

            noise_pred = torch.cat([noise_pred_ref, noise_pred_tgt], dim=0)

            # compute the previous noisy sample x_t -> x_t-1
            latents = ddim_sampler.step(noise_pred, t_i, latents)["prev_sample"]
            # latents = style_aligned_next_step(ddim_sampler, noise_pred, t_i, latents)
    # decode the latents to images
    imgs = sd.decode_latents(latents)

    # save the images
    for i, img in enumerate(imgs):
        save_image(img, f"{out_path}/img_{i}.png")
        print(f"saved {out_path}/img_{i}.png")
