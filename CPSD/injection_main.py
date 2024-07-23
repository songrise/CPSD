# %%
import argparse, os

os.environ["HF_HOME"] = "/root/autodl-tmp/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/.cache"

import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    DiffusionPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.image_processor import VaeImageProcessor
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
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


def _get_text_embeddings(prompt: str, tokenizer, text_encoder, device):
    # Tokenize text and get embeddings
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    with torch.no_grad():
        prompt_embeds = text_encoder(
            text_input_ids.to(device),
            output_hidden_states=True,
        )

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    if prompt == "":
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        return negative_prompt_embeds, negative_pooled_prompt_embeds
    return prompt_embeds, pooled_prompt_embeds


def _encode_text_sdxl(model: StableDiffusionXLPipeline, prompt: str):
    device = model._execution_device
    (
        prompt_embeds,
        pooled_prompt_embeds,
    ) = _get_text_embeddings(prompt, model.tokenizer, model.text_encoder, device)
    (
        prompt_embeds_2,
        pooled_prompt_embeds_2,
    ) = _get_text_embeddings(prompt, model.tokenizer_2, model.text_encoder_2, device)
    prompt_embeds = torch.cat((prompt_embeds, prompt_embeds_2), dim=-1)
    text_encoder_projection_dim = model.text_encoder_2.config.projection_dim
    add_time_ids = model._get_add_time_ids(
        (1024, 1024), (0, 0), (1024, 1024), torch.float16, text_encoder_projection_dim
    ).to(device)
    # repeat the time ids for each prompt
    add_time_ids = add_time_ids.repeat(len(prompt), 1)
    added_cond_kwargs = {
        "text_embeds": pooled_prompt_embeds_2,
        "time_ids": add_time_ids,
    }
    return added_cond_kwargs, prompt_embeds


def _encode_text_sdxl_with_negative(
    model: StableDiffusionXLPipeline, prompt: List[str]
):

    B = len(prompt)
    added_cond_kwargs, prompt_embeds = _encode_text_sdxl(model, prompt)
    added_cond_kwargs_uncond, prompt_embeds_uncond = _encode_text_sdxl(
        model, ["" for _ in range(B)]
    )
    prompt_embeds = torch.cat(
        (
            prompt_embeds_uncond,
            prompt_embeds,
        )
    )
    added_cond_kwargs = {
        "text_embeds": torch.cat(
            (added_cond_kwargs_uncond["text_embeds"], added_cond_kwargs["text_embeds"])
        ),
        "time_ids": torch.cat(
            (added_cond_kwargs_uncond["time_ids"], added_cond_kwargs["time_ids"])
        ),
    }
    return added_cond_kwargs, prompt_embeds


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
    if isinstance(pipe, StableDiffusionPipeline):
        text_embeddings = pipe._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )
        added_cond_kwargs = None
    elif isinstance(pipe, StableDiffusionXLPipeline):
        added_cond_kwargs, text_embeddings = _encode_text_sdxl_with_negative(
            pipe, prompt
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
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
            added_cond_kwargs=added_cond_kwargs,
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
    vae_decoder = VaeImageProcessor(vae_scale_factor=pipe.vae.config.scaling_factor)
    # Encode prompt
    if isinstance(pipe, StableDiffusionPipeline):
        text_embeddings = pipe._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )
        added_cond_kwargs = None
    elif isinstance(pipe, StableDiffusionXLPipeline):
        added_cond_kwargs, text_embeddings = _encode_text_sdxl_with_negative(
            pipe, prompt
        )

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    # save

    # Create a random starting point if we don't have one already
    # if start_latents is None:
    # TODO Jun 26: check scale
    latent_shape = (
        (1, 4, 64, 64) if isinstance(pipe, StableDiffusionPipeline) else (1, 4, 64, 64)
    )
    generative_latent = torch.randn(latent_shape, device=device)
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
    #     if do_classifier_free_guxidance:
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
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Post-processing
        # images = vae_decoder.postprocess(latents)
    pipe.vae.to(dtype=torch.float32)
    latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
    latents = 1 / pipe.vae.config.scaling_factor * latents
    images = pipe.vae.decode(latents, return_dict=False)[0]
    images = (images / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    images = pipe.numpy_to_pil(images)
    if isinstance(pipe, StableDiffusionXLPipeline):
        pipe.vae.to(dtype=torch.float16)
    # images = pipe.decode_latents(latents)
    # images = pipe.numpy_to_pil(images)

    return images


# Sample function (regular DDIM), but disentangle the content and style
@torch.no_grad()
def sample_disentangled_reference(
    pipe,
    prompt,
    start_step=0,
    start_latents=None,
    intermediate_latents_content=None,
    intermediate_latents_style=None,
    guidance_scale=3.5,
    num_inference_steps=30,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    use_content_anchor=True,
    negative_prompt="",
    device=device,
):
    negative_prompt = [""] * len(prompt)
    vae_decoder = VaeImageProcessor(vae_scale_factor=pipe.vae.config.scaling_factor)
    # Encode prompt
    if isinstance(pipe, StableDiffusionPipeline):
        text_embeddings = pipe._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )
        added_cond_kwargs = None
    elif isinstance(pipe, StableDiffusionXLPipeline):
        added_cond_kwargs, text_embeddings = _encode_text_sdxl_with_negative(
            pipe, prompt
        )

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    # save

    # Create a random starting point if we don't have one already
    # if start_latents is None:
    # TODO Jun 26: check scale
    latent_shape = (
        (1, 4, 64, 64) if isinstance(pipe, StableDiffusionPipeline) else (1, 4, 64, 64)
    )

    latents = start_latents.clone()

    latents = latents.repeat(len(prompt), 1, 1, 1)

    # assume that the first latent is used for reconstruction
    for i in tqdm(range(start_step, num_inference_steps)):

        if use_content_anchor:
            latents[0] = intermediate_latents_content[(-i + 1)]
        latents[1] = intermediate_latents_style[(-i + 1)]

        t = pipe.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Post-processing
        # images = vae_decoder.postprocess(latents)
    pipe.vae.to(dtype=torch.float32)
    latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
    latents = 1 / pipe.vae.config.scaling_factor * latents
    images = pipe.vae.decode(latents, return_dict=False)[0]
    images = (images / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    images = pipe.numpy_to_pil(images)
    if isinstance(pipe, StableDiffusionXLPipeline):
        pipe.vae.to(dtype=torch.float16)
    # images = pipe.decode_latents(latents)
    # images = pipe.numpy_to_pil(images)

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
    if isinstance(pipe, StableDiffusionPipeline):
        text_embeddings = pipe._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )
        added_cond_kwargs = None
        latents = start_latents.clone().detach()
    elif isinstance(pipe, StableDiffusionXLPipeline):
        added_cond_kwargs, text_embeddings = _encode_text_sdxl_with_negative(
            pipe, [prompt]
        )  # Latents are now the specified start latents
        latents = start_latents.clone().detach().half()

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
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
            added_cond_kwargs=added_cond_kwargs,
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
    share_cross_attn=False,
    share_resnet_layers=[0, 1],
    share_attn_layers=[],
    c2s_layers=[0, 1],
    share_key=True,
    share_query=True,
    share_value=False,
    use_adain=True,
    use_content_anchor=True,
    output_dir: str = None,
    resnet_mode: str = None,
    return_intermediate=False,
    intermediate_latents=None,
):
    with torch.no_grad():
        pipe.vae.to(dtype=torch.float32)
        latent = pipe.vae.encode(input_image.to(device) * 2 - 1)
        # latent = pipe.vae.encode(input_image.to(device))
        l = pipe.vae.config.scaling_factor * latent.latent_dist.sample()
        if isinstance(pipe, StableDiffusionXLPipeline):
            pipe.vae.to(dtype=torch.float16)
    if intermediate_latents is None:
        inverted_latents = invert(
            pipe, l, input_image_prompt, num_inference_steps=num_steps
        )
    else:
        inverted_latents = intermediate_latents

    attn_injection.register_attention_processors(
        pipe,
        base_dir=output_dir,
        resnet_mode=resnet_mode,
        attn_mode="disentangled_pnp" if disentangle else "pnp",
        disentangle=disentangle,
        share_resblock=True,
        share_attn=share_attn,
        share_cross_attn=share_cross_attn,
        share_resnet_layers=share_resnet_layers,
        share_attn_layers=share_attn_layers,
        share_key=share_key,
        share_query=share_query,
        share_value=share_value,
        use_adain=use_adain,
        c2s_layers=c2s_layers,
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
        pipe,
        unset_share_attn=True,
        unset_share_resblock=True,
    )
    if return_intermediate:
        return final_im, inverted_latents
    return final_im


def style_image_with_inversion_reference(
    pipe,
    input_image,
    input_image_prompt,
    style_image,
    num_steps=100,
    start_step=30,
    guidance_scale=3.5,
    disentangle=False,
    share_attn=False,
    share_cross_attn=False,
    share_resnet_layers=[0, 1],
    share_attn_layers=[],
    share_key=True,
    share_query=True,
    share_value=False,
    use_adain=True,
    use_content_anchor=True,
    output_dir: str = None,
    resnet_mode: str = None,
):
    with torch.no_grad():
        pipe.vae.to(dtype=torch.float32)
        latent = pipe.vae.encode(input_image.to(device) * 2 - 1)
        style_latent = pipe.vae.encode(style_image.to(device) * 2 - 1)
        # latent = pipe.vae.encode(input_image.to(device))
        l = pipe.vae.config.scaling_factor * latent.latent_dist.sample()
        l_s = pipe.vae.config.scaling_factor * style_latent.latent_dist.sample()
        if isinstance(pipe, StableDiffusionXLPipeline):
            pipe.vae.to(dtype=torch.float16)

    inverted_latents_content = invert(
        pipe, l, input_image_prompt, num_inference_steps=num_steps
    )

    inverted_latents_style = invert(
        pipe, l_s, input_image_prompt, num_inference_steps=num_steps
    )

    attn_injection.register_attention_processors(
        pipe,
        base_dir=output_dir,
        resnet_mode=resnet_mode,
        attn_mode="disentangled_pnp" if disentangle else "pnp",
        disentangle=disentangle,
        share_resblock=True,
        share_attn=share_attn,
        share_cross_attn=share_cross_attn,
        share_resnet_layers=share_resnet_layers,
        share_attn_layers=share_attn_layers,
        share_key=share_key,
        share_query=share_query,
        share_value=share_value,
        use_adain=use_adain,
    )

    if disentangle:
        final_im = sample_disentangled_reference(
            pipe,
            ["" for _ in range(3)],
            start_latents=inverted_latents_content[-(start_step + 1)][None],
            intermediate_latents_content=inverted_latents_content,
            intermediate_latents_style=inverted_latents_style,
            start_step=start_step,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            use_content_anchor=use_content_anchor,
        )
    else:
        final_im = sample(
            pipe,
            # style_prompt,
            start_latents=inverted_latents_content[-(start_step + 1)][None],
            intermediate_latents=inverted_latents_content,
            start_step=start_step,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
        )

    # unset the attention processors
    attn_injection.unset_attention_processors(
        pipe,
        unset_share_attn=True,
        unset_share_resblock=True,
    )

    return final_im


if __name__ == "__main__":

    # Load a pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base"
    ).to(device)

    # pipe = DiffusionPipeline.from_pretrained(
    #     # "playgroundai/playground-v2-1024px-aesthetic",
    #     torch_dtype=torch.float16,
    #     use_safetensors=True,
    #     add_watermarker=False,
    #     variant="fp16",
    # )
    # pipe.to("cuda")

    # Set up a DDIM scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    parser = argparse.ArgumentParser(description="Stable Diffusion with OmegaConf")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file"
    )
    # mode = "single_control_content"
    mode = "text"
    if mode == "text":
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
            resolution = 512 if isinstance(pipe, StableDiffusionXLPipeline) else 512
            input_image = utils.exp_utils.get_processed_image(
                image_path, device, resolution
            )

            prompt_in = [
                src_prompt,  # reconstruction
                tgt_prompt,  # uncontrolled style
                tgt_prompt,  # controlled style
                # "",
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
                resnet_mode=cfg.resnet_mode,
                share_attn=cfg.share_attn,
                share_cross_attn=cfg.share_cross_attn,
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
    elif mode == "image_style":
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
            resolution = 512 if isinstance(pipe, StableDiffusionXLPipeline) else 512
            input_image = utils.exp_utils.get_processed_image(
                image_path, device, resolution
            )
            style_image = utils.exp_utils.get_processed_image(
                "/root/autodl-tmp/data/style/starryNight.png", device, resolution
            )

            imgs = style_image_with_inversion_reference(
                pipe,
                input_image,
                src_prompt,
                style_image,
                num_steps=cfg.num_steps,
                start_step=cfg.start_step,
                guidance_scale=cfg.style_cfg_scale,
                disentangle=cfg.disentangle,
                resnet_mode=cfg.resnet_mode,
                share_attn=cfg.share_attn,
                share_cross_attn=cfg.share_cross_attn,
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
    ##########TEMP DEBUGGING
    elif mode == "single_control_content":
        anno_dir = "/root/autodl-tmp/data/ideogram/annotation_control.json"
        anno = json.load(open(anno_dir))
        for img_id in range(1, 65):
            # img_id =
            # if img_id != 3:
            #     continue
            image_path = anno[img_id - 1]["image_path"]
            # image_path = "/root/autodl-tmp/budin2.png"
            src_prompt = ""
            tgt_prompt = anno[img_id - 1]["target_prompt"]
            input_image = utils.exp_utils.get_processed_image(image_path, device, 512)
            prompt_in = [
                src_prompt,  # reconstruction
                tgt_prompt,  # uncontrolled style
                tgt_prompt,  # controlled style
            ]
            # cumulative from 4 to 11
            content_control_layers = [
                [0],
                [0, 1],
                [0, 1, 2],
                [0, 1, 2, 3],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6, 7],
                [0, 1, 2, 3, 4, 5, 6, 7, 8],
            ]
            # for resnet_mode in ["hidden"]:
            for resnet_mode in ["hidden", "pnp"]:
                crt_xts = None
                for i, resnet_layer in enumerate(content_control_layers):
                    imgs, xts = style_image_with_inversion(
                        pipe,
                        input_image,
                        src_prompt,
                        style_prompt=prompt_in,
                        num_steps=50,
                        start_step=5,
                        guidance_scale=7.5,
                        disentangle=True,
                        resnet_mode=resnet_mode,
                        share_attn=True,
                        share_cross_attn=True,
                        share_resnet_layers=resnet_layer,
                        share_attn_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                        share_key=True,
                        share_query=True,
                        share_value=False,
                        use_adain=True,
                        use_content_anchor=True,
                        output_dir=".",
                        return_intermediate=True,
                        intermediate_latents=crt_xts,
                    )
                    crt_xts = xts
                    temp_save_path = f"/root/autodl-tmp/CPSD/vis_out/fine_grained_control/{img_id}/{resnet_mode}"

                    # temp_save_path = f"/root/autodl-tmp/CPSD/vis_out/fine_grained_control_c2s/{img_id}/{resnet_mode}"
                    if not os.path.exists(temp_save_path):
                        os.makedirs(temp_save_path)
                    # save
                    result_img = imgs[2]
                    result_img.save(
                        f"{temp_save_path}/fine_control_{img_id}_{len(resnet_layer)}.png"
                    )
                    print(
                        f"Image saved as {temp_save_path}/fine_control_{img_id}_{len(resnet_layer)}.png"
                    )
    elif mode == "single_control_style":
        img_id = 62
        image_path = f"/root/autodl-tmp/data/ideogram/{img_id}.png"
        tgt_prompt = "A B&W pencil sketch, detailed cross-hatching"
        input_image = utils.exp_utils.get_processed_image(image_path, device, 512)
        prompt_in = [
            "",  # reconstruction
            tgt_prompt,  # uncontrolled style
            "",  # controlled style
        ]

        for c2s_layers in [
            [],
            [0, 1],
            [0, 1, 2, 3],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [6, 7, 8],
        ]:
            tau = 5
            inject_style = True
            # for inject_style in [True, False]:
            # share_cross = inject_style
            imgs = style_image_with_inversion(
                pipe,
                input_image,
                "",
                style_prompt=prompt_in,
                num_steps=50,
                start_step=tau,
                guidance_scale=7.5,
                disentangle=True,
                resnet_mode="hidden",
                share_attn=True,
                share_cross_attn=True,
                share_resnet_layers=[0, 1, 2, 3],
                share_attn_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                share_key=True,
                share_query=True,
                share_value=False,
                use_adain=True,
                use_content_anchor=True,
                output_dir=".",
                c2s_layers=c2s_layers,
            )
            temp_save_path = f"/root/autodl-tmp/CPSD/vis_out/c2s_control/{img_id}/style"
            if not os.path.exists(temp_save_path):
                os.makedirs(temp_save_path)
            # save
            result_img_style = imgs[1]
            result_img = imgs[2]
            result_img_style.save(
                f"{temp_save_path}/style_{img_id}_{c2s_layers}_{tau}.png"
            )
            result_img.save(
                f"{temp_save_path}/c2s_control_{img_id}_{c2s_layers}_{tau}.png"
            )
            print(
                f"Image saved as {temp_save_path}/c2s_control_{img_id}_{c2s_layers}_{tau}.png"
            )
    elif mode == "tau":
        img_id = 0
        image_path = f"/root/autodl-tmp/data/standard/horse.png"
        src_prompt = ""
        tgt_prompt = "A Fauvism painting"
        input_image = utils.exp_utils.get_processed_image(image_path, device, 512)
        prompt_in = [
            src_prompt,  # reconstruction
            tgt_prompt,  # uncontrolled style
            tgt_prompt,  # controlled style
        ]

        for start_step in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]:
            inject_style = True
            # for inject_style in [True, False]:
            # share_cross = inject_style
            imgs = style_image_with_inversion(
                pipe,
                input_image,
                src_prompt,
                style_prompt=prompt_in,
                num_steps=50,
                start_step=start_step,
                guidance_scale=7.5,
                disentangle=True,
                resnet_mode="hidden",
                share_attn=False,
                share_cross_attn=False,
                share_resnet_layers=[],
                share_attn_layers=[],
                share_key=False,
                share_query=False,
                share_value=False,
                use_adain=True,
                use_content_anchor=True,
                output_dir=".",
            )
            temp_save_path = f"/root/autodl-tmp/CPSD/vis_out/tau"
            if not os.path.exists(temp_save_path):
                os.makedirs(temp_save_path)
            # save
            result_img = imgs[2]
            result_img.save(f"{temp_save_path}/tau_{img_id}_{start_step}.png")
            print(f"Image saved as {temp_save_path}/tau_{img_id}_{start_step}.png")
    elif mode == "explore":
        # randomly sample a img id, randomly sample a style prompt
        import random

        while True:
            annotation = json.load(
                open("/root/autodl-tmp/data/ideogram/annotation.json")
            )
            # img_id = random.randint(1, 73)
            img_id = [74, 75, 76, 77, 78][random.randint(0, 4)]
            image_path = f"/root/autodl-tmp/CPSD/data/horse.png"
            src_prompt = ""
            # style_id = [4, 6, 24, 69, 64, 30, 31, 18, 14, 62, 35, 33, 44, 43][
            #     random.randint(0, 12)
            # ]
            temp_save_path = f"/root/autodl-tmp/CPSD/vis_out/cfg/horse"
            style_id = random.randint(1, 72)
            # temp_save_path = f"/root/autodl-tmp/CPSD/vis_out/explore/{img_id}"
            # if the style image already exists, skip
            if os.path.exists(f"{temp_save_path}/style_{style_id}.png"):
                continue
            tgt_prompt = annotation[style_id]["target_prompt"]
            input_image = utils.exp_utils.get_processed_image(image_path, device, 512)
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
                num_steps=50,
                start_step=0,
                guidance_scale=7.5,
                disentangle=True,
                resnet_mode="hidden",
                share_attn=True,
                share_cross_attn=True,
                share_resnet_layers=[0, 1, 2, 3],
                share_attn_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                share_key=True,
                share_query=True,
                share_value=False,
                use_adain=True,
                use_content_anchor=True,
                output_dir=".",
            )

            if not os.path.exists(temp_save_path):
                os.makedirs(temp_save_path)
            # save
            result_img = imgs[2]
            result_img.save(f"{temp_save_path}/style_{style_id}.png")
            print(f"{temp_save_path}/style_{style_id}.png")
    elif mode == "cfg":
        # randomly sample a img id, randomly sample a style prompt
        import random

        while True:
            annotation = json.load(
                open("/root/autodl-tmp/data/ideogram/annotation.json")
            )
            img_id = 13

            image_path = f"/root/autodl-tmp/data/misc/trump.png"
            src_prompt = ""
            # style_id = [4, 6, 24, 69, 64, 30, 31, 18, 14, 62, 35, 33, 44, 43][
            #     random.randint(0, 12)
            # ]
            temp_save_path = f"/root/autodl-tmp/CPSD/vis_out/cfg/trump"
            # if the style image already exists, skip
            for cfg in [3.5, 5.0, 7.5, 10.0]:
                for tgt_prompt in [
                    # "Expressionist painting",
                    "Expressionist style painting by Edvard Munch, abstract.",
                    # "Fauvism style painting, bold brush stroke",
                    # "Expressionist style painting, detailed brush stroke, warm colors",
                ]:
                    input_image = utils.exp_utils.get_processed_image(
                        image_path, device, 512
                    )
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
                        num_steps=50,
                        start_step=0,
                        guidance_scale=cfg,
                        disentangle=True,
                        resnet_mode="hidden",
                        share_attn=True,
                        share_cross_attn=True,
                        share_resnet_layers=[0, 1, 2, 3],
                        share_attn_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                        share_key=True,
                        share_query=True,
                        share_value=False,
                        use_adain=True,
                        use_content_anchor=True,
                        output_dir=".",
                    )

                    if not os.path.exists(temp_save_path):
                        os.makedirs(temp_save_path)
                    # save
                    result_img = imgs[2]
                    result_img.save(f"{temp_save_path}/{cfg}_{tgt_prompt}.png")
                    print(f"{temp_save_path}/{cfg}_{tgt_prompt}.png")
