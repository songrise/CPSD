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

import omegaconf
import utils.exp_utils
import json

device = torch.device("cuda")


# Sample function (regular DDIM)
@torch.no_grad()
def sample(
    pipe,
    prompt,
    start_step=0,
    start_latents=None,
    intermediate_latents=None,
    guidance_scale=3.5,
    num_inference_steps=30,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=device,
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
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    latents = latents.repeat(len(prompt), 1, 1, 1)
    # assume that the first latent is used for reconstruction
    for i in tqdm(range(start_step, num_inference_steps)):
        latents[0] = intermediate_latents[(-i + 1)]
        t = pipe.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        # Normally we'd rely on the scheduler to handle the update step:
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Instead, let's do it ourselves:
        # prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
        # alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        # alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        # predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        # direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        # latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt

    # Post-processing
    images = pipe.decode_latents(latents)
    images = pipe.numpy_to_pil(images)

    return images


# Sample function (regular DDIM), but disentangle the content and style
@torch.no_grad()
def sample_disentangled(
    pipe,
    prompt,
    start_step=0,
    start_latents=None,
    intermediate_latents=None,
    guidance_scale=3.5,
    num_inference_steps=30,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    use_content_anchor=True,
    negative_prompt="",
    device=device,
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
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    # if start_latents is None:
    generative_latent = torch.randn(1, 4, 64, 64, device=device)
    generative_latent *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    latents = latents.repeat(len(prompt), 1, 1, 1)
    # randomly initalize the 1st lantent for generation

    # TODO Jun 18: here, when use generative latent, should first denoise it for start_step times
    # for i in tqdm(range(0, start_step)):
    #     t = pipe.scheduler.timesteps[i]

    #     # Expand the latents if we are doing classifier free guidance
    #     latent_model_input = (
    #         torch.cat([generative_latent] * 2)
    #         if do_classifier_free_guidance
    #         else generative_latent
    #     )
    #     latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

    #     # Predict the noise residual
    #     noise_pred = pipe.unet(
    #         latent_model_input, t, encoder_hidden_states=text_embeddings
    #     ).sample

    #     # Perform guidance
    #     if do_classifier_free_guidance:
    #         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #         noise_pred = noise_pred_uncond + guidance_scale * (
    #             noise_pred_text - noise_pred_uncond
    #         )

    #     generative_latent = pipe.scheduler.step(
    #         noise_pred, t, generative_latent
    #     ).prev_sample

    latents[1] = generative_latent
    # assume that the first latent is used for reconstruction
    for i in tqdm(range(start_step, num_inference_steps)):

        if use_content_anchor:
            latents[0] = intermediate_latents[(-i + 1)]
        t = pipe.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
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

    # Post-processing
    images = pipe.decode_latents(latents)
    images = pipe.numpy_to_pil(images)

    return images


## Inversion
@torch.no_grad()
def invert(
    pipe,
    start_latents,
    prompt,
    guidance_scale=3.5,
    num_inference_steps=50,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=device,
):

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
    )

    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (
            alpha_t_next.sqrt() / alpha_t.sqrt()
        ) + (1 - alpha_t_next).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)


def style_image_with_inversion(
    pipe,
    input_image,
    input_image_prompt,
    style_prompt,
    num_steps=100,
    start_step=30,
    guidance_scale=3.5,
    disentangle=False,
    share_attn=False,
    share_resnet_layers=[0, 1],
    share_attn_layers=[],
    share_key=True,
    share_query=True,
    share_value=False,
    use_adain=True,
    use_content_anchor=True,
    output_dir: str = None,
):
    with torch.no_grad():
        latent = pipe.vae.encode(input_image.to(device) * 2 - 1)
        # latent = pipe.vae.encode(input_image.to(device))
    l = 0.18215 * latent.latent_dist.sample()
    inverted_latents = invert(
        pipe, l, input_image_prompt, num_inference_steps=num_steps
    )

    attn_injection.register_attention_processors(
        pipe.unet,
        base_dir=output_dir,
        resnet_mode="disentangle" if disentangle else "default",
        attn_mode="disentangled_pnp" if disentangle else "pnp",
        share_resblock=True,
        share_attn=share_attn,
        share_resnet_layers=share_resnet_layers,
        share_attn_layers=share_attn_layers,
        share_key=share_key,
        share_query=share_query,
        share_value=share_value,
        use_adain=use_adain,
    )

    if disentangle:
        final_im = sample_disentangled(
            pipe,
            style_prompt,
            start_latents=inverted_latents[-(start_step + 1)][None],
            intermediate_latents=inverted_latents,
            start_step=start_step,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            use_content_anchor=use_content_anchor,
        )
    else:
        final_im = sample(
            pipe,
            style_prompt,
            start_latents=inverted_latents[-(start_step + 1)][None],
            intermediate_latents=inverted_latents,
            start_step=start_step,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
        )

    # unset the attention processors
    attn_injection.unset_attention_processors(
        pipe.unet,
        unset_share_attn=True,
        unset_share_resblock=True,
    )

    return final_im


if __name__ == "__main__":

    # Load a pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base"
    ).to(device)

    # Set up a DDIM scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    parser = argparse.ArgumentParser(description="Stable Diffusion with OmegaConf")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file"
    )
    args = parser.parse_args()
    config_dir = args.config
    cfg = OmegaConf.load(config_dir)

    base_output_path = cfg.out_path
    if not os.path.exists(cfg.out_path):
        os.makedirs(cfg.out_path)
    base_output_path = os.path.join(base_output_path, cfg.exp_name)

    experiment_output_path = utils.exp_utils.make_unique_experiment_path(
        base_output_path
    )

    # Save the experiment configuration
    config_file_path = os.path.join(experiment_output_path, "config.yaml")
    omegaconf.OmegaConf.save(cfg, config_file_path)

    # Seed all

    annotation = json.load(open(cfg.annotation))
    with open(os.path.join(experiment_output_path, "annotation.json"), "w") as f:
        json.dump(annotation, f)
    for i, entry in enumerate(annotation):
        utils.exp_utils.seed_all(cfg.seed)
        image_path = entry["image_path"]
        src_prompt = entry["source_prompt"]
        tgt_prompt = entry["target_prompt"]

        input_image = utils.exp_utils.get_processed_image(image_path, device)

        prompt_in = [
            src_prompt,  # reconstruction
            tgt_prompt,  # uncontrolled style
            tgt_prompt,  # controlled style
        ]

        imgs = style_image_with_inversion(
            pipe,
            input_image,
            src_prompt,
            style_prompt=prompt_in,
            num_steps=cfg.num_steps,
            start_step=cfg.start_step,
            guidance_scale=cfg.style_cfg_scale,
            disentangle=cfg.disentangle,
            share_attn=cfg.share_attn,
            share_resnet_layers=cfg.share_resnet_layers,
            share_attn_layers=cfg.share_attn_layers,
            share_key=cfg.share_key,
            share_query=cfg.share_query,
            share_value=cfg.share_value,
            use_content_anchor=cfg.use_content_anchor,
            use_adain=cfg.use_adain,
            output_dir=experiment_output_path,
        )

        for j, img in enumerate(imgs):
            img.save(f"{experiment_output_path}/out_{i}_{j}.png")
            print(f"Image saved as {experiment_output_path}/out_{i}_{j}.png")
