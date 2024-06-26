# -*- coding : utf-8 -*-
# @FileName  : diffusion.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Oct 20, 2022
# @Github    : https://github.com/songrise
# @Description: Dream Fusion loss, implementation from https://github.com/ashawkey/stable-dreamfusion


from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler
from tqdm import tqdm

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import models.attn_injection as attn_injection


class StableDiffusion(nn.Module):
    """
    Warper class for the SDS loss based on the stable diffusion model.
    """

    def __init__(self, device, version):
        super().__init__()

        try:
            with open("./TOKEN", "r") as f:
                self.token = f.read().replace("\n", "")  # remove the last \n!
                print(f"[INFO] loaded hugging face access token from ./TOKEN!")
        except FileNotFoundError as e:
            self.token = False
            print(
                f"[INFO] try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`."
            )

        self.sd_version = version
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)
        self.use_depth = False
        if self.sd_version == "1.4":
            self.model_key = "CompVis/stable-diffusion-v1-4"
        elif self.sd_version == "1.5":
            self.model_key = "runwayml/stable-diffusion-v1-5"
        elif self.sd_version == "2.0":
            self.use_depth = True
            # self.model_key = '/root/autodl-tmp/local_models/sd-v2'
            self.model_key = "stabilityai/stable-diffusion-2"
        elif self.sd_version == "2.1":
            self.use_depth = True
            self.model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == "xl":
            self.use_depth = True
            self.model_key = "stabilityai/stable-diffusion-xl-base-1.0"
        print(f"[INFO] loading stable diffusion {self.sd_version} ...")

        # 1. Load the autoencoder model which will be used to decode the latents into image space.
        self.vae = AutoencoderKL.from_pretrained(
            self.model_key, subfolder="vae", use_safetensors=True
        ).to(self.device)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_key, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_key, subfolder="text_encoder", use_safetensors=True
        ).to(self.device)

        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_key, subfolder="unet", use_safetensors=True
        ).to(self.device)

        # 4. Create a scheduler for inference
        self.scheduler = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=self.num_train_timesteps,
        )
        # self.scheduler = DDIMScheduler(num_train_timesteps=self.num_train_timesteps)

        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        print(f"[INFO] loaded stable diffusion!")

    def config_ddim_scheduler(self):
        self.scheduler = DDIMScheduler(num_train_timesteps=self.num_train_timesteps)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience
        print(f"[INFO] configured DDIM scheduler!")

    def get_text_embeds(self, prompt: list):
        if not isinstance(prompt, list):
            prompt = [prompt]
        # Tokenize text and get embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(
            [""] * len(prompt),
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def manual_backward(
        self,
        text_embeddings,
        pred_rgb: torch.Tensor,
        guidance_scale=100,
        pred_depth: torch.Tensor = None,
        latent=None,
    ) -> None:
        """
        backward the SDS loss the the pred_rgb.
        Input:
            pred_rgb: Tensor, [1, 3, H, W] assume requires grad

        return:
            grad_map: [1, 3, H, W], in the same dimension.
        """
        if latent is None:
            h, w = pred_rgb.shape[-2:]

            # zero pad to 512x512
            pred_rgb_512 = torchvision.transforms.functional.pad(
                pred_rgb, ((512 - w) // 2,), fill=0, padding_mode="constant"
            )
            pred_rgb_512 = F.interpolate(
                pred_rgb, (512, 512), mode="bilinear", align_corners=False
            )
            # debug_utils.dump_tensor(pred_rgb_512, 'pred_rgb_512.pkl')
            if self.use_depth and pred_depth is not None:
                pred_depth = F.interpolate(
                    pred_depth, size=(64, 64), mode="bicubic", align_corners=False
                )
                pred_depth = (
                    2.0
                    * (pred_depth - pred_depth.min())
                    / (pred_depth.max() - pred_depth.min())
                    - 1.0
                )
                pred_depth = torch.cat([pred_depth] * 2)
            # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

            # encode image into latents with vae, requires grad!
            # _t = time.time()
            latents = self.encode_imgs(pred_rgb_512)
        else:
            latents = latent

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device
        )
        # t = torch.randint(self.min_step, 100, [1], dtype=torch.long, device=self.device)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            if self.use_depth and pred_depth is not None:
                latent_model_input = torch.cat([latent_model_input, pred_depth], dim=1)
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        # CFG
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # w(t), sigma_t^2
        w = 1 - self.alphas[t]
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])

        grad = w * (noise_pred - noise)

        # clip grad for stable training?
        # grad = grad.clamp(-1, 1)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        # _t = time.time()
        latents.backward(gradient=grad, retain_graph=True)

    def manual_backward_nfsd(
        self,
        text_embeddings,
        pred_rgb: torch.Tensor,
        guidance_scale=100,
        pred_depth: torch.Tensor = None,
        latent=None,
    ) -> None:
        """
        backward the SDS loss the the pred_rgb.
        Input:
            pred_rgb: Tensor, [1, 3, H, W] assume requires grad

        return:
            grad_map: [1, 3, H, W], in the same dimension.
        """
        if latent is None:
            h, w = pred_rgb.shape[-2:]

            # zero pad to 512x512
            pred_rgb_512 = torchvision.transforms.functional.pad(
                pred_rgb, ((512 - w) // 2,), fill=0, padding_mode="constant"
            )
            pred_rgb_512 = F.interpolate(
                pred_rgb, (512, 512), mode="bilinear", align_corners=False
            )
            # debug_utils.dump_tensor(pred_rgb_512, 'pred_rgb_512.pkl')
            if self.use_depth and pred_depth is not None:
                pred_depth = F.interpolate(
                    pred_depth, size=(64, 64), mode="bicubic", align_corners=False
                )
                pred_depth = (
                    2.0
                    * (pred_depth - pred_depth.min())
                    / (pred_depth.max() - pred_depth.min())
                    - 1.0
                )
                pred_depth = torch.cat([pred_depth] * 2)
            # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

            # encode image into latents with vae, requires grad!
            # _t = time.time()
            latents = self.encode_imgs(pred_rgb_512)
        else:
            latents = latent

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device
        )
        # t = torch.randint(self.min_step, 100, [1], dtype=torch.long, device=self.device)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            if self.use_depth and pred_depth is not None:
                latent_model_input = torch.cat([latent_model_input, pred_depth], dim=1)
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # for the negative text
            noise_ = torch.randn_like(latents)
            text_neg = "unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, dark, low-resolution, gloomy"
            text_neg_embeddings = self.get_text_embeds(text_neg)
            noise_pred_neg = self.unet(
                latent_model_input, t, encoder_hidden_states=text_neg_embeddings
            ).sample
            noise_pred_neg_uncod, noise_pred_neg_text = noise_pred_neg.chunk(2)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        # CFG
        if t > 200:
            noise_pred = (noise_pred_uncond - noise_pred_neg_text) + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        else:
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        # w(t), sigma_t^2
        w = 1 - self.alphas[t]
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])

        # noise free
        grad = w * noise_pred

        # clip grad for stable training?
        # grad = grad.clamp(-1, 1)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        # _t = time.time()
        latents.backward(gradient=grad, retain_graph=True)

    def manual_backward_csd(
        self,
        text_embeddings,
        pred_rgb: torch.Tensor,
        guidance_scale=100,
        pred_depth: torch.Tensor = None,
        latent=None,
    ) -> None:
        """
        backward the SDS loss the the pred_rgb.
        Input:
            pred_rgb: Tensor, [1, 3, H, W] assume requires grad

        return:
            grad_map: [1, 3, H, W], in the same dimension.
        """
        if latent is None:
            h, w = pred_rgb.shape[-2:]

            # zero pad to 512x512
            pred_rgb_512 = torchvision.transforms.functional.pad(
                pred_rgb, ((512 - w) // 2,), fill=0, padding_mode="constant"
            )
            pred_rgb_512 = F.interpolate(
                pred_rgb, (512, 512), mode="bilinear", align_corners=False
            )
            # debug_utils.dump_tensor(pred_rgb_512, 'pred_rgb_512.pkl')
            if self.use_depth and pred_depth is not None:
                pred_depth = F.interpolate(
                    pred_depth, size=(64, 64), mode="bicubic", align_corners=False
                )
                pred_depth = (
                    2.0
                    * (pred_depth - pred_depth.min())
                    / (pred_depth.max() - pred_depth.min())
                    - 1.0
                )
                pred_depth = torch.cat([pred_depth] * 2)
            # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

            # encode image into latents with vae, requires grad!
            # _t = time.time()
            latents = self.encode_imgs(pred_rgb_512)
        else:
            latents = latent

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device
        )
        # t = torch.randint(self.min_step, 100, [1], dtype=torch.long, device=self.device)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            if self.use_depth and pred_depth is not None:
                latent_model_input = torch.cat([latent_model_input, pred_depth], dim=1)
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        # CFG, but only use the guidance part, discard the uncond term
        noise_pred = guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = 1 - self.alphas[t]
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])

        # also noise free
        grad = w * (noise_pred)

        # clip grad for stable training?
        # grad = grad.clamp(-1, 1)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        # _t = time.time()
        latents.backward(gradient=grad, retain_graph=True)

    def manual_backward_dds(
        self,
        src_text_embd,
        src_img,
        tgt_text_embd,
        tgt_img,
        guidance_scale=100,
        src_latent=None,
        tgt_latent=None,
    ):
        """
        calculate and mannually backward the gradient derived
        from the delta denoising score by one step.

        Return: the calculated score (gradient)
        """

        # h, w = src_img.shape[-2:]
        # zero pad to 512x512
        # pred_rgb_512 = torchvision.transforms.functional.pad(src_img, ((512-w)//2, ), fill=0, padding_mode='constant')
        if src_latent is None:
            src_img_512 = F.interpolate(
                src_img, (512, 512), mode="bilinear", align_corners=False
            )
            src_latent = self.encode_imgs(src_img_512).requires_grad_(False)
        if tgt_latent is None:
            tgt_img_512 = F.interpolate(
                tgt_img, (512, 512), mode="bilinear", align_corners=False
            )
            tgt_latent = self.encode_imgs(tgt_img_512).requires_grad_(True)

        t = torch.randint(
            self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device
        )

        # noise_overwrite = torch.rand_like(src_latent)
        noise_overwrite = None
        with torch.no_grad():  # notice calc_grad does not rely on autograd
            src_grad = self.calc_grad(
                src_text_embd,
                src_img,
                guidance_scale=guidance_scale,
                latent=src_latent,
                t_overwrite=t,
                noise_overwrite=noise_overwrite,
            )

        tgt_grad = self.calc_grad(
            tgt_text_embd,
            tgt_img,
            guidance_scale=guidance_scale,
            latent=tgt_latent,
            t_overwrite=t,
            noise_overwrite=noise_overwrite,
        )
        grad = tgt_grad - src_grad
        tgt_latent.backward(gradient=grad, retain_graph=True)
        return grad.detach()

    def calc_grad(
        self,
        text_embeddings,
        pred_rgb: torch.Tensor,
        guidance_scale=100,
        latent=None,
        t_overwrite=None,
        noise_overwrite=None,
    ) -> torch.Tensor:
        """
        calculate the gradient of the predicted rgb
        Input:
            pred_rgb: Tensor, [1, 3, H, W] assume requires grad

        return:
            grad_map: [1, 3, H, W], in the same dimension.
        """
        # interp to 512x512 to be fed into vae.

        # _t = time.time()
        if latent is None:
            h, w = pred_rgb.shape[-2:]

            # zero pad to 512x512
            pred_rgb_512 = torchvision.transforms.functional.pad(
                pred_rgb, ((512 - w) // 2,), fill=0, padding_mode="constant"
            )
            pred_rgb_512 = F.interpolate(
                pred_rgb, (512, 512), mode="bilinear", align_corners=False
            )
            # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

            # encode image into latents with vae, requires grad!
            # _t = time.time()
            latents = self.encode_imgs(pred_rgb_512)
        else:
            latents = latent

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if t_overwrite is not None:
            if not isinstance(t_overwrite, torch.Tensor):
                t_overwrite = torch.tensor(
                    t_overwrite, device=self.device, dtype=torch.long
                )
            t = t_overwrite
        else:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [1],
                dtype=torch.long,
                device=self.device,
            )
        # t = torch.randint(self.min_step, 50, [1], dtype=torch.long, device=self.device)

        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            if noise_overwrite is not None:
                noise = noise_overwrite
            else:
                noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # w(t), sigma_t^2
        w = 1 - self.alphas[t]
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        # clip grad for stable training?
        # grad = grad.clamp(-1, 1)
        #!HARDCODED Dec 06: modified for getting latent grad
        # # manually backward, since we omitted an item in grad and cannot simply autodiff.
        # # _t = time.time()
        # latents.backward(gradient=grad, retain_graph=True)
        # # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')

        # pred_rgb_grad = pred_rgb.grad.detach().clone()
        # return pred_rgb_grad
        #!HARDCODED Dec 06: new
        # TODO Dec 06: refactor the latent flag
        if latent is None:
            latents.backward(gradient=grad, retain_graph=True)
            # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')

            pred_rgb_grad = pred_rgb.grad.detach().clone()
            return pred_rgb_grad
        else:
            return grad

    def calc_grad_injected(
        self,
        src_text_embed,
        tgt_text_embed,
        latents: torch.Tensor,
        t_i,  # index of the time step
        intermediate_latents: torch.Tensor,
        guidance_scale: float = 100,
        inference_steps: int = 50,
        noise_overwrite: torch.Tensor = None,
        disentangle: bool = False,
    ) -> torch.Tensor:
        """
        calculate the gradient of the predicted rgb
        Input:
            pred_rgb: Tensor, [1, 3, H, W] assume requires grad

        return:
            grad_map: [1, 3, H, W], in the same dimension.
        """
        # interp to 512x512 to be fed into vae.

        # _t = time.time()
        assert isinstance(
            self.scheduler, DDIMScheduler
        ), "DDIMScheduler is required for DDIM inverse."

        self.scheduler.set_timesteps(inference_steps)
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if t_i is None:
            raise ValueError("t_i is required for DDIM inversion.")
            # t = torch.randint(
            #     self.min_step,
            #     self.max_step + 1,
            #     [1],
            #     dtype=torch.long,
            #     device=self.device,
            # )

        # _t = time.time()
        with torch.no_grad():
            # add noise
            if noise_overwrite is not None:
                # if shape not same, then repeat
                if noise.shape != latents.shape:
                    B = latents.shape[0]
                    noise = noise_overwrite.repeat(B, 1, 1, 1)
                else:
                    noise = noise_overwrite
            else:
                noise = torch.randn_like(latents)

            latents_noisy = intermediate_latents[t_i]  # from DDIM inversion
            reversed_timesteps = list(reversed(self.scheduler.timesteps))
            t = reversed_timesteps[t_i + 1]
            # t = self.scheduler.timesteps[t_i]
            tgt_latents_noisy = self.scheduler.add_noise(latents, noise, t)
            if disentangle:
                latents_noisy = latents_noisy.repeat(3, 1, 1, 1)
            else:
                latents_noisy = latents_noisy.repeat(2, 1, 1, 1)

            # pred noise
            # TODO here, only the first should be from ddim
            # still wrong here is that latents is un-noised, but should be noised
            latents_noisy[1] = tgt_latents_noisy
            latent_model_input = torch.cat([latents_noisy] * 2)
            text_embed = torch.cat([src_text_embed, tgt_text_embed], dim=0)  # [4, ...]
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embed
            ).sample
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # w(t), sigma_t^2
        w = 1 - self.alphas[t]
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        # TODO Jun 07: here should not be noise refer to HIFA.... also, the w is incorrect.

        grad = w * (noise_pred - noise)
        # grad = w * (noise_pred)

        # clip grad for stable training?
        # grad = grad.clamp(-1, 1)
        # extract the grad of the stylization term
        grad = grad[-1, ...].unsqueeze(0)

        if latents is None:
            latents.backward(gradient=grad, retain_graph=True)
            # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')

            pred_rgb_grad = pred_rgb.grad.detach().clone()
            return pred_rgb_grad
        else:
            return grad

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100):
        """
        Use sd to perform one step update of the model.
        """

        # interp to 512x512 to be fed into vae.

        # _t = time.time()
        pred_rgb_512 = F.interpolate(
            pred_rgb, (512, 512), mode="bilinear", align_corners=False
        )
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device
        )
        # t = torch.randint(500,501 [1], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        # _t = time.time()
        latents = self.encode_imgs(pred_rgb_512)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # w(t), sigma_t^2
        w = 1 - self.alphas[t]
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        # clip grad for stable training?
        # grad = grad.clamp(-1, 1)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        # _t = time.time()
        latents.backward(gradient=grad, retain_graph=True)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')

        return 0  # dummy loss value

    def produce_latents(
        self,
        text_embeddings,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):

        if latents is None:
            latents = torch.randn(
                (
                    text_embeddings.shape[0] // 2,
                    self.unet.in_channels,
                    height // 8,
                    width // 8,
                ),
                device=self.device,
            )

        self.scheduler.set_timesteps(num_inference_steps)

        # with torch.autocast('cuda'):
        with torch.cuda.amp.autocast():
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    )["sample"]

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]

        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample
            # imgs = self.vae.decode(latents)

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(
        self,
        prompts,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        depth=None,
        latents=None,
        return_torch=True,
        debug_dump=False,
    ):

        if isinstance(prompts, str):
            prompts = [prompts]
        if debug_dump:
            attn_injection.register_attention_processors(
                self.unet,
                "style_aligned",
                "/root/autodl-tmp/CPSD/attn/debug",
            )
        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts)  # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(
            text_embeds,
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]
        if return_torch:
            return imgs
        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs

    @torch.no_grad()
    def ddim_inverse(
        self,
        start_latents,
        prompt,
        guidance_scale=3.5,
        num_inference_steps=80,
        use_cfg=True,
        negative_prompt="",
        device="cuda",
    ):
        """
        run ddim inversion on the given latents and prompt
        return all intermidiate latents produced by DDIM inversion.
        """
        assert isinstance(
            self.scheduler, DDIMScheduler
        ), "DDIMScheduler is required for DDIM inverse."

        # Encode prompt
        text_embeddings = self.get_text_embeds(prompt)
        # Latents are now the specified start latents
        latents = start_latents.clone()

        # We'll keep a list of the inverted latents as the process goes on
        intermediate_latents = []

        # Set num inference steps
        # TODO Apr 13:  check if this would affect the distillation
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
        timesteps = reversed(self.scheduler.timesteps)

        for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

            # We'll skip the final iteration
            if i >= num_inference_steps - 1:
                continue

            t = timesteps[i]

            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if use_cfg else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # Perform guidance
            if use_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
            next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
            alpha_t = self.alphas[current_t]
            alpha_t_next = self.alphas[next_t]

            # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
            latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (
                alpha_t_next.sqrt() / alpha_t.sqrt()
            ) + (1 - alpha_t_next).sqrt() * noise_pred

            # Store
            intermediate_latents.append(latents)

        return torch.cat(intermediate_latents), timesteps


# if __name__ == '__main__':
#     import os
#     import numpy as np

#     def fix_randomness(seed=42):

#         # random.seed(seed)
#         # torch.backends.cudnn.deterministic = True
#         # torch.backends.cudnn.benchmark = False
#         # torch.manual_seed(seed)
#         # np.random.seed(seed)

#         # https: // www.zhihu.com/question/542479848/answer/2567626957
#         os.environ['PYTHONHASHSEED'] = str(seed)

#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)

#         np.random.seed(seed)

#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#         torch.backends.cudnn.enabled = False
#     import argparse
#     import matplotlib.pyplot as plt
#     import PIL.Image as Image


#     depth = torch.zeros((1, 1, 64, 64))

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--prompt', type=str, default = "a photo of a cute corgi")
#     parser.add_argument('-H', type=int, default=512)
#     parser.add_argument('-W', type=int, default=512)
#     parser.add_argument('--steps', type=int, default=100)
#     opt = parser.parse_args()

#     device = torch.device('cuda')

#     sd = StableDiffusion(device,version="2.0")
#     # prompt_ = "front view of the face of Captain Marvel, photorealistic style."
#     # prompt_ = "front view of the body of the Hulk wearing blue jeans, photorealistic style."
#     prompt_ = "a starry night painting"
#     imgs = []
#     # fix_randomness(42)

#     for _ in range(2):
#         img = sd.prompt_to_img(prompt_, 512, 512, opt.steps)
#         img = img / 255.
#         imgs.append(torch.from_numpy(img))
#         print("done one")
#     imgs = torch.cat(imgs, dim=0)
#     # save image as a grid
#     imgs = imgs.permute(0, 3, 1, 2)
#     img_grid = torchvision.utils.make_grid(imgs, nrow = 5, padding = 10)
#     torchvision.utils.save_image(img_grid, 'img_grid.png')
#     print('Image saved as img_grid.png')

if __name__ == "__main__":

    def fix_randomness(seed=42):
        import os
        import numpy as np

        # https: // www.zhihu.com/question/542479848/answer/2567626957
        os.environ["PYTHONHASHSEED"] = str(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        np.random.seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    def zero_shot_gen_sds():
        # optimize 2d tensor as image representation.
        import PIL.Image as Image

        device = torch.device("cuda")
        # image = nn.Parameter(torch.empty(1, 3, 512, 512, device=device))
        # #init with the image to be edited
        # with open("panda_snow.png", 'rb') as f:
        #     image_ = Image.open(f)
        #     image_ = torchvision.transforms.functional.resize(image_, (512, 512))
        #     image_ = torchvision.transforms.functional.to_tensor(image_)[:3,...].unsqueeze(0)
        #     image.data = image_.to(device)

        latent = nn.Parameter(torch.empty(1, 4, 64, 64, device=device))
        latent.data = torch.randn_like(latent.data)
        optimizer = torch.optim.SGD([latent], lr=1.0)
        decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
        n_iter = 200
        sd = StableDiffusion(device, version="2.0")
        image = sd.decode_latents(latent)
        prompt = "a Squirrel is snowboarding"
        image_steps = []
        for i in range(n_iter):
            optimizer.zero_grad()
            sd.manual_backward(
                sd.get_text_embeds(prompt), image, guidance_scale=20, latent=latent
            )
            optimizer.step()
            if i % 20 == 0:
                decay.step()
                print(f"[INFO] iter {i}, loss {torch.norm(latent.grad)}")
                image = sd.decode_latents(latent)
                image_steps.append(image.detach().clone())

        # visualize as grid
        image_steps = torch.cat(image_steps, dim=0)
        from torchvision.utils import make_grid

        grid = make_grid(image_steps, nrow=5, padding=10)
        # save
        from torchvision.utils import save_image

        save_image(grid, "image_steps.png")

    def edit_sds():
        # optimize 2d tensor as image representation.
        import PIL.Image as Image

        device = torch.device("cuda")

        sd = StableDiffusion(device, version="2.0")
        latent = nn.Parameter(torch.empty(1, 4, 64, 64, device=device))
        # init with the image to be edited
        with open(
            "/root/autodl-tmp/gaussian-splatting/data/fangzhou/images/000001.png", "rb"
        ) as f:
            image_ = Image.open(f)
            image_ = torchvision.transforms.functional.resize(image_, (512, 512))
            image_ = torchvision.transforms.functional.to_tensor(image_)[
                :3, ...
            ].unsqueeze(0)
            image_latent = sd.encode_imgs(image_.to(device))
            latent.data = image_latent.data
        optimizer = torch.optim.SGD([latent], lr=1.0)
        decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
        n_iter = 200

        image = sd.decode_latents(latent)
        prompt = "van gogh potrait"
        image_steps = []
        for i in range(n_iter):
            optimizer.zero_grad()
            sd.manual_backward(
                sd.get_text_embeds(prompt), image, guidance_scale=20, latent=latent
            )
            optimizer.step()
            if i % 20 == 0:
                decay.step()
                print(f"[INFO] iter {i}, loss {torch.norm(latent.grad)}")
                image = sd.decode_latents(latent)
                image_steps.append(image.detach().clone())

        # visualize as grid
        image_steps = torch.cat(image_steps, dim=0)
        from torchvision.utils import make_grid

        grid = make_grid(image_steps, nrow=5, padding=10)
        # save
        from torchvision.utils import save_image

        save_image(grid, "image_steps.png")

    def edit_dds():
        # optimize 2d tensor as image representation.
        import PIL.Image as Image

        device = torch.device("cuda")

        sd = StableDiffusion(device, version="2.0")
        latent = nn.Parameter(torch.empty(1, 4, 64, 64, device=device))
        # init with the image to be edited
        with open("/root/autodl-tmp/CPSD/data/girl_1.png", "rb") as f:
            image_ = Image.open(f)
            image_ = torchvision.transforms.functional.resize(image_, (512, 512))
            image_ = torchvision.transforms.functional.to_tensor(image_)[
                :3, ...
            ].unsqueeze(0)
            image_latent = sd.encode_imgs(image_.to(device))
            latent.data = image_latent.data
            latent_orig = latent.data.clone().requires_grad_(False)
        optimizer = torch.optim.SGD([latent], lr=1.0)
        decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
        n_iter = 200

        image = sd.decode_latents(latent)
        prompt_a = "a photo of a girl"
        prompt_b = "a photo of Barack Obama"
        image_steps = []
        for i in range(n_iter):
            optimizer.zero_grad()
            grad_a = sd.calc_grad(
                sd.get_text_embeds(prompt_a),
                image,
                guidance_scale=20,
                latent=latent_orig,
            )
            grad_b = sd.calc_grad(
                sd.get_text_embeds(prompt_b), image, guidance_scale=20, latent=latent
            )
            grad_apply = grad_b - grad_a
            latent.backward(gradient=grad_apply, retain_graph=True)
            optimizer.step()
            if i % 20 == 0:
                decay.step()
                print(f"[INFO] iter {i}, loss {torch.norm(latent.grad)}")
                image = sd.decode_latents(latent)
                image_steps.append(image.detach().clone())

        # visualize as grid
        image_steps = torch.cat(image_steps, dim=0)
        from torchvision.utils import make_grid

        grid = make_grid(image_steps, nrow=5, padding=10)
        # save
        from torchvision.utils import save_image

        save_image(grid, "image_steps.png")

    fix_randomness()
    edit_sds()
    # edit_dds()
