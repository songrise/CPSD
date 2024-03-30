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
import uuid
import omegaconf


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


def first_ddim_inversion(
    sd, ddim_sampler, src_img_512, text_embeddings, num_inference_steps
):
    z0 = sd.encode_imgs(src_img_512)
    latent = z0.clone().detach()
    zts = [z0]

    for i in tqdm.trange(num_inference_steps):
        t_i = ddim_sampler.timesteps[num_inference_steps - i - 1]
        t_i = torch.tensor(t_i, device=src_img_512.device, dtype=torch.long)
        latent_model_input = torch.cat([latent] * 2)

        with torch.no_grad():
            noise_pred = sd.unet(
                latent_model_input, t_i, encoder_hidden_states=text_embeddings[:2, ...]
            )["sample"]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 2 * (noise_pred_text - noise_pred_uncond)
            latent = style_aligned_next_step(
                ddim_sampler, noise_pred, t_i, latent, num_inference_steps
            )
            zts.append(latent)

    zts = torch.cat(zts).flip(0)
    return zts


@torch.no_grad()
def stylize_loop_pnp(
    sd,
    cfg_scale,
    ddim_sampler,
    latents,
    zts,
    text_embeddings,
    offset,
    tau_attn,
    tau_feat,
):

    with torch.cuda.amp.autocast():
        for i in tqdm.trange(ddim_sampler.num_inference_steps - offset):
            t_i = ddim_sampler.timesteps[i + offset]
            t_i = torch.tensor(t_i, device=zts.device, dtype=torch.long)
            if (ddim_sampler.num_inference_steps - i) == tau_attn:
                attn_injection.unset_attention_processors(
                    sd.unet, unset_share_attn=True
                )
            if (ddim_sampler.num_inference_steps - i) == tau_feat:
                attn_injection.unset_attention_processors(
                    sd.unet, unset_share_resblock=True
                )

            latents[0] = zts[i]  # [ref, tgt]
            latent_model_input = torch.repeat_interleave(latents, 2, dim=0)  #
            # latent_model_input = torch.cat([latents] * 2, dim=0)

            with torch.no_grad():
                noise_pred = sd.unet(
                    latent_model_input, t_i, encoder_hidden_states=text_embeddings
                )["sample"]

            (
                noise_pred_uncond_ref,
                noise_pred_text_ref,
                noise_pred_uncond_tgt,
                noise_pred_text_tgt,
            ) = noise_pred.chunk(4)

            noise_pred_ref = noise_pred_uncond_ref + cfg_scale * (
                noise_pred_text_ref - noise_pred_uncond_ref
            )
            noise_pred_tgt = noise_pred_uncond_tgt + cfg_scale * (
                noise_pred_text_tgt - noise_pred_uncond_tgt
            )

            noise_pred = torch.cat([noise_pred_ref, noise_pred_tgt], dim=0)
            latents = ddim_sampler.step(noise_pred, t_i, latents)["prev_sample"]

    return latents


@torch.no_grad()
def stylize_loop_si(
    sd,
    cfg_scale,
    ddim_sampler_content,
    ddim_sampler_style,
    latents,
    zts,
    text_embeddings,
    offset,
    tau_attn,
    tau_feat,
):

    with torch.cuda.amp.autocast():
        for i in tqdm.trange(ddim_sampler_content.num_inference_steps - offset):
            t_i = ddim_sampler_content.timesteps[i + offset]
            t_i = torch.tensor(t_i, device=zts.device, dtype=torch.long)
            if (ddim_sampler_content.num_inference_steps - i) == tau_attn:
                attn_injection.unset_attention_processors(
                    sd.unet, unset_share_attn=True
                )
            if (ddim_sampler_content.num_inference_steps - i) == tau_feat:
                attn_injection.unset_attention_processors(
                    sd.unet, unset_share_resblock=True
                )

            latents[0] = zts[i]  # [ref, tgt]
            latent_model_input = torch.repeat_interleave(latents, 2, dim=0)  #
            # latent_model_input = torch.cat([latents] * 2, dim=0)

            with torch.no_grad():
                noise_pred = sd.unet(
                    latent_model_input, t_i, encoder_hidden_states=text_embeddings
                )["sample"]

            (
                noise_pred_uncond_style,
                noise_pred_text_style,
                noise_pred_uncond_content,
                noise_pred_text_content,
            ) = noise_pred.chunk(4)

            noise_pred_style = noise_pred_uncond_style + cfg_scale * (
                noise_pred_text_style - noise_pred_uncond_style
            )
            noise_pred_content = noise_pred_uncond_content + cfg_scale * (
                noise_pred_text_content - noise_pred_uncond_content
            )

            noise_pred = torch.cat([noise_pred_style, noise_pred_content], dim=0)

            latent_style = latents[1].unsqueeze(0)
            latent_style = ddim_sampler_style.step(noise_pred[:1], t_i, latent_style)[
                "prev_sample"
            ]
            latent_content = latents[0].unsqueeze(0)
            latent_content = ddim_sampler_content.step(
                noise_pred[1:], t_i, latent_content
            )["prev_sample"]
            # latents = ddim_sampler_content.step(noise_pred, t_i, latents)["prev_sample"]
            # latents = torch.cat([latent_style, latent_content])
    return latents


def main_pnp(config_dir):

    cfg = OmegaConf.load(config_dir)

    device = torch.device("cuda")
    sd = StableDiffusion(device, version="2.1")

    g_cpu = torch.Generator(device="cpu")
    g_cpu.manual_seed(cfg.seed)

    base_output_path = cfg.out_path
    base_output_path = os.path.join(base_output_path, cfg.exp_name)
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path)
    # Create a folder with a unique name for each experiment according to the process uid
    experiment_id = str(uuid.uuid4())[
        :8
    ]  # Generate a unique identifier for the experiment
    experiment_output_path = os.path.join(base_output_path, experiment_id)
    os.makedirs(experiment_output_path)

    # Save the experiment configuration
    config_file_path = os.path.join(experiment_output_path, "config.yaml")
    omegaconf.OmegaConf.save(cfg, config_file_path)

    if not os.path.exists(cfg.out_path):
        os.makedirs(cfg.out_path)

    src_img = Image.open(cfg.src_img)
    src_img = transforms.ToTensor()(src_img).unsqueeze(0).to(device)
    src_prompt = cfg.src_prompt
    src_text_embedding = sd.get_text_embeds(src_prompt)

    ddim_sampler = DDIMScheduler(num_train_timesteps=1000)
    ddim_sampler.set_timesteps(cfg.num_steps)

    h, w = src_img.shape[-2:]
    src_img_512 = torchvision.transforms.functional.pad(
        src_img, ((512 - w) // 2,), fill=0, padding_mode="constant"
    )
    src_img_512 = F.interpolate(
        src_img, (512, 512), mode="bilinear", align_corners=False
    )

    if src_img_512.shape[1] == 4:
        src_img_512 = src_img_512[:, :3]

    zts = first_ddim_inversion(
        sd, ddim_sampler, src_img_512, src_text_embedding, cfg.num_steps
    )

    offset = 0
    tgt_prompts: List[str] = cfg.tgt_prompt

    batch_size = cfg.batch_size
    num_batches = (len(tgt_prompts) + batch_size - 1) // batch_size
    # construct the starting point of DDIM forward
    zt = zts[offset].unsqueeze(0)

    # latents = torch.rand((batch_size, 4, 64, 64), device=zts.device)
    # latents[0] = zt
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(tgt_prompts))
        tgt_prompts_batch = tgt_prompts[start_idx:end_idx]

        attn_injection.register_attention_processors(
            sd.unet,
            base_dir="/root/autodl-tmp/CPSD/attn/debug",
            mode="pnp",
            share_resblock=True if len(cfg.share_resnet_layers) > 0 else False,
            share_attn=True if len(cfg.share_attn_layers) > 0 else False,
            # share_attn=False,
            share_resnet_layers=cfg.share_resnet_layers,
            share_attn_layers=cfg.share_attn_layers,
            share_key=cfg.share_key,
            share_query=cfg.share_query,
            share_value=cfg.share_value,
        )
        # tgt_text_embedding = sd.get_text_embeds(tgt_prompts_batch)
        # style_text_embedding = torch.cat(
        #     [src_text_embedding, tgt_text_embedding], dim=0
        # )
        style_prompt = [src_prompt] + tgt_prompts_batch
        style_text_embedding = sd.get_text_embeds(style_prompt)

        latents = torch.cat([zt] * 2)
        latents = stylize_loop_pnp(
            sd,
            cfg.style_cfg_scale,
            ddim_sampler,
            latents,
            zts.detach().clone(),
            style_text_embedding,
            offset,
            cfg.tau_attn,
            cfg.tau_feat,
        )

        imgs = sd.decode_latents(latents)

        for tgt_prompt in tgt_prompts_batch:
            tgt_name = (
                tgt_prompt.replace(" ", "_").replace(",", "").replace(".", "").lower()
            )
            save_image(imgs[1], f"{experiment_output_path}/{tgt_name}.png")
            print(f"saved {experiment_output_path}/{tgt_name}.png")


def main_si(config_dir):

    cfg = OmegaConf.load(config_dir)

    device = torch.device("cuda")
    sd = StableDiffusion(device, version="2.1")

    g_cpu = torch.Generator(device="cpu")
    g_cpu.manual_seed(cfg.seed)

    base_output_path = cfg.out_path
    base_output_path = os.path.join(base_output_path, cfg.exp_name)
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path)
    # Create a folder with a unique name for each experiment according to the process uid
    experiment_id = str(uuid.uuid4())[
        :8
    ]  # Generate a unique identifier for the experiment
    experiment_output_path = os.path.join(base_output_path, experiment_id)
    os.makedirs(experiment_output_path)

    # Save the experiment configuration
    config_file_path = os.path.join(experiment_output_path, "config.yaml")
    omegaconf.OmegaConf.save(cfg, config_file_path)

    if not os.path.exists(cfg.out_path):
        os.makedirs(cfg.out_path)

    src_img = Image.open(cfg.src_img)  # content image
    src_img = transforms.ToTensor()(src_img).unsqueeze(0).to(device)
    src_prompt = cfg.src_prompt
    src_text_embedding = sd.get_text_embeds(src_prompt)

    ddim_sampler_content = DDIMScheduler(num_train_timesteps=1000)
    ddim_sampler_content.set_timesteps(cfg.num_steps)

    h, w = src_img.shape[-2:]
    src_img_512 = torchvision.transforms.functional.pad(
        src_img, ((512 - w) // 2,), fill=0, padding_mode="constant"
    )
    src_img_512 = F.interpolate(
        src_img, (512, 512), mode="bilinear", align_corners=False
    )

    if src_img_512.shape[1] == 4:
        src_img_512 = src_img_512[:, :3]

    zts_c = first_ddim_inversion(
        sd, ddim_sampler_content, src_img_512, src_text_embedding, cfg.num_steps
    )

    if False:  # decode and save the image at each ddim step
        all_imgs = []
        for i, latent in enumerate(zts_c):
            img = sd.decode_latents(latent.unsqueeze(0))
            all_imgs.append(img)
        # grid
        grid = make_grid(torch.cat(all_imgs), nrow=8)
        # grid = torchvision.transforms.functional.resize(grid)
        save_image(grid, f"{experiment_output_path}/ddim_c_grid.png")

    # ddim inverse the style image
    ddim_sampler_style = DDIMScheduler(num_train_timesteps=1000)
    ddim_sampler_style.set_timesteps(cfg.num_steps)
    # ddim_sampler_style = ddim_sampler_content
    style_image = Image.open(cfg.style_img)  # style image
    style_image = transforms.ToTensor()(style_image).unsqueeze(0).to(device)
    style_prompt = cfg.style_prompt
    style_text_embedding = sd.get_text_embeds(style_prompt)
    h, w = style_image.shape[-2:]
    style_image_512 = torchvision.transforms.functional.pad(
        style_image, ((512 - w) // 2,), fill=0, padding_mode="constant"
    )
    style_image_512 = F.interpolate(
        style_image, (512, 512), mode="bilinear", align_corners=False
    )
    if style_image_512.shape[1] == 4:
        style_image_512 = style_image_512[:, :3]
    zts_s = first_ddim_inversion(
        sd, ddim_sampler_style, style_image_512, style_text_embedding, cfg.num_steps
    )

    offset = 0
    tgt_prompts: List[str] = cfg.tgt_prompt

    batch_size = cfg.batch_size
    num_batches = (len(tgt_prompts) + batch_size - 1) // batch_size

    zt_c = zts_c[offset].unsqueeze(0)
    zt_s = zts_s[offset].unsqueeze(0)
    latents = torch.cat([zt_s, zt_c])  # inject the feature of style to content
    # latents = torch.rand((batch_size, 4, 64, 64), device=zts.device)
    # latents[0] = zt

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(tgt_prompts))
        tgt_prompts_batch = tgt_prompts[start_idx:end_idx]

        # attn_injection.register_attention_processors(
        #     sd.unet,
        #     base_dir="/root/autodl-tmp/CPSD/attn/debug",
        #     mode="pnp",
        #     share_resblock=True if len(cfg.share_resnet_layers) > 0 else False,
        #     share_attn=True if len(cfg.share_attn_layers) > 0 else False,
        #     share_resnet_layers=cfg.share_resnet_layers,
        #     share_attn_layers=cfg.share_attn_layers,
        #     share_key=cfg.share_key,
        #     share_query=cfg.share_query,
        #     share_value=cfg.share_value,
        # )
        # tgt_text_embedding = sd.get_text_embeds(tgt_prompts_batch)
        # style_text_embedding = torch.cat(
        #     [src_text_embedding, tgt_text_embedding], dim=0
        # )
        style_prompt = [src_prompt] + tgt_prompts_batch
        style_text_embedding = sd.get_text_embeds(style_prompt)
        latents = stylize_loop_si(
            sd,
            cfg.style_cfg_scale,
            ddim_sampler_content,
            ddim_sampler_style,
            latents,
            zts_s.detach().clone(),
            style_text_embedding,
            offset,
            cfg.tau_attn,
            cfg.tau_feat,
        )

        imgs = sd.decode_latents(latents)
        # save denoised style image
        save_image(imgs[0], f"{experiment_output_path}/style.png")

        # save the generated images
        for tgt_prompt in tgt_prompts_batch:
            tgt_name = (
                tgt_prompt.replace(" ", "_").replace(",", "").replace(".", "").lower()
            )
            save_image(imgs[1], f"{experiment_output_path}/{tgt_name}.png")
            print(f"saved {experiment_output_path}/{tgt_name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion with OmegaConf")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file"
    )
    parser.add_argument(
        "--method", type=str, default="pnp", help="Method to run: pnp or si"
    )
    args = parser.parse_args()
    config_dir = args.config
    if args.method == "pnp":
        main_pnp(config_dir)
    elif args.method == "si":
        main_si(config_dir)
