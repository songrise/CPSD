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
import glob
import models.attn_injection as attn_injection
import random


def parse_args():
    parser = argparse.ArgumentParser(description="2D Experiment")
    parser.add_argument(
        "--mode",
        type=str,
        default="edit",
        help="generation or edit",
        choices=["generation", "edit", "stylize"],
    )
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--method", type=str, default="sds", help="sds, nfsd, csd")
    parser.add_argument("--output_dir", type=str, default="output/debug/")

    return parser.parse_args()


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


def register_injection(sd):
    attn_injection.register_attention_processors(
        sd.unet,
        base_dir=None,
        resnet_mode="default",
        attn_mode="pnp",
        share_resblock=True,
        share_attn=True,
        share_resnet_layers=[0, 1, 2, 3],
        share_attn_layers=[0, 1, 2, 3],
        share_key=True,
        share_query=True,
        share_value=False,
        use_adain=False,
    )


def unset_injection(sd):
    attn_injection.unset_attention_processors(sd.unet, True, True)


def zero_shot_gen(
    sd_model, tgt_prompt: str, method: str, save_path: str = "image_steps.png", **kwargs
):
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
    if method == "sds":
        guidance_scale = 100
    elif method == "nfsd":
        guidance_scale = 7.5
    elif method == "csd":
        guidance_scale = 7.5
    latent = nn.Parameter(torch.empty(1, 4, 64, 64, device=device))
    latent.data = torch.randn_like(latent.data)
    optimizer = torch.optim.SGD([latent], lr=1)
    # decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
    n_iter = 1000
    image = sd_model.decode_latents(latent)
    image_steps = []
    embed = sd_model.get_text_embeds(tgt_prompt)
    for i in range(n_iter):
        optimizer.zero_grad()
        if method == "sds":
            sd_model.manual_backward(
                embed, image, guidance_scale=guidance_scale, latent=latent
            )
        elif method == "nfsd":
            sd_model.manual_backward_nfsd(
                embed, image, guidance_scale=guidance_scale, latent=latent
            )
        elif method == "csd":
            sd_model.manual_backward_csd(
                embed, image, guidance_scale=guidance_scale, latent=latent
            )
        optimizer.step()
        if i % 200 == 0:
            # decay.step()
            print(f"[INFO] iter {i}, loss {torch.norm(latent.grad)}")
            image = sd_model.decode_latents(latent)
            image_steps.append(image.detach().clone())

    # visualize as grid
    image_steps = torch.cat(image_steps, dim=0)
    from torchvision.utils import make_grid

    grid = make_grid(image_steps, nrow=5, padding=10)
    # save
    save_image(grid, save_path)


def edit_sds(sd, latent, prompt, image_path, method="sds"):
    # optimize 2d tensor as image representation.

    lr = 0.2 if method == "sds" else 1.0
    optimizer = torch.optim.SGD([latent], lr=lr)
    decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
    n_iter = 100

    image = sd.decode_latents(latent)
    image_steps = []
    embed = sd.get_text_embeds(prompt)
    for i in range(n_iter):
        optimizer.zero_grad()
        if method == "sds":
            sd.manual_backward(embed, image, guidance_scale=100, latent=latent)
        elif method == "nfsd":
            sd.manual_backward_nfsd(embed, image, guidance_scale=7.5, latent=latent)
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

    save_image(grid, image_path)
    print(f"Done, Image saved as {image_path}")


def edit_nfsd(sd, latent, prompt, image_path):
    return edit_sds(sd, latent, prompt, image_path, method="nfsd")


def edit_dds(sd, latent, prompt_a, prompt_b, image_path):
    latent_orig = latent.data.clone().requires_grad_(False)
    optimizer = torch.optim.SGD([latent], lr=1.0)
    decay = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.9)
    n_iter = 200

    image = sd.decode_latents(latent)
    image_steps = []
    embed_a = sd.get_text_embeds(prompt_a)
    embed_b = sd.get_text_embeds(prompt_b)
    for i in range(n_iter):
        optimizer.zero_grad()
        # grad_a = sd.calc_grad(sd.get_text_embeds(prompt_a), image, guidance_scale=7.5, latent=latent_orig)
        # grad_b = sd.calc_grad(sd.get_text_embeds(prompt_b), image, guidance_scale=7.5, latent=latent)
        # grad_apply = grad_b - grad_a
        # latent.backward(gradient=grad_apply, retain_graph=True)
        sd.manual_backward_dds(
            embed_a,
            image,
            embed_b,
            image,
            guidance_scale=7.5,
            src_latent=latent_orig,
            tgt_latent=latent,
        )
        optimizer.step()

        if i % 40 == 0:
            print(f"[INFO] iter {i}, loss {torch.norm(latent.grad)}")
            image = sd.decode_latents(latent)
            image_steps.append(image.detach().clone())
            # decay.step()

    # visualize as grid
    image_steps = torch.cat(image_steps, dim=0)

    grid = make_grid(image_steps, nrow=5, padding=10)
    # save

    save_image(grid, image_path)
    print(f"Done, Image saved as {image_path}")


def edit_cpsd(
    sd: StableDiffusion,
    latent: torch.Tensor,
    src_text_prompt,
    tgt_text_prompt,
    image_path,
):
    latent_orig = latent.data.clone().requires_grad_(False)
    optimizer = torch.optim.SGD([latent], lr=0.8)
    decay = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.9)
    n_iter = 200

    image = sd.decode_latents(latent)
    image_steps = []
    src_text_embed = sd.get_text_embeds(src_text_prompt)
    tgt_text_embed = sd.get_text_embeds(tgt_text_prompt)
    sd.config_ddim_scheduler()
    inverted_latents, sampled_time_steps = sd.ddim_inverse(
        start_latents=latent_orig,
        prompt=src_text_prompt,
        guidance_scale=5,
        num_inference_steps=50,
    )

    for i in range(n_iter):
        optimizer.zero_grad()
        # grad_a = sd.calc_grad(sd.get_text_embeds(prompt_a), image, guidance_scale=7.5, latent=latent_orig)
        # grad_b = sd.calc_grad(sd.get_text_embeds(prompt_b), image, guidance_scale=7.5, latent=latent)
        # grad_apply = grad_b - grad_a
        # latent.backward(gradient=grad_apply, retain_graph=True)
        t_i = random.randint(0, len(inverted_latents) - 1)
        register_injection(sd)
        tgt_score = sd.calc_grad_injected(
            src_text_embed,
            tgt_text_embed,
            latents=latent,
            t_i=t_i,
            guidance_scale=7.5,
            # src_latent=latent_orig,
            intermediate_latents=inverted_latents,
        )
        unset_injection(sd)
        src_score = sd.calc_grad(
            src_text_embed, image, guidance_scale=7.5, latent=latent, t_overwrite=t_i
        )
        # grad = tgt_score - src_score
        grad = tgt_score
        latent.backward(gradient=grad, retain_graph=True)
        optimizer.step()

        if i % 40 == 0:
            print(f"[INFO] iter {i}, loss {torch.norm(latent.grad)}")
            image = sd.decode_latents(latent)
            image_steps.append(image.detach().clone())
            # decay.step()

    # visualize as grid
    image_steps = torch.cat(image_steps, dim=0)

    grid = make_grid(image_steps, nrow=5, padding=10)
    # save

    save_image(grid, image_path)
    print(f"Done, Image saved as {image_path}")


def sd_forward(prompt):
    # prompt_ = "front view of the body of the Hulk wearing blue jeans, photorealistic style."

    imgs = []
    # fix_randomness(42)
    device = torch.device("cuda")
    sd = StableDiffusion(device, version="2.1")
    latent = nn.Parameter(torch.empty(1, 4, 64, 64, device=device))
    for _ in range(8):
        img = sd.prompt_to_img(prompt, 512, 512, 100)
        img = img / 255.0
        imgs.append(torch.from_numpy(img))
        print("done one")
    imgs = torch.cat(imgs, dim=0)
    # save image as a grid
    imgs = imgs.permute(0, 3, 1, 2)
    img_grid = torchvision.utils.make_grid(imgs, nrow=5, padding=10)
    torchvision.utils.save_image(img_grid, "img_grid.png")
    print("Image saved as img_grid.png")


if __name__ == "__main__":

    # edit_sds()
    # edit_dds()
    def generation_experiment(sd_model, out_dir: str = None, tgt_prompt: str = None):
        print("start generation experiment")
        # prompt = "A painting of an astronaut riding a horse on the moon."
        # tgt_prompt = "a zoomed out DSLR photo of a panda wearing a chef's hat and kneading bread dough on a countertop"
        # prompt = "“a photo of a asian man in pixar style”"

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        zero_shot_gen(
            sd_model,
            tgt_prompt,
            method="sds",
            save_path=f"{out_dir}/image_steps_sds.png",
        )
        zero_shot_gen(
            sd_model,
            tgt_prompt,
            method="nfsd",
            save_path=f"{out_dir}/image_steps_nfsd.png",
        )
        zero_shot_gen(
            sd_model,
            tgt_prompt,
            method="csd",
            save_path=f"{out_dir}/image_steps_csd.png",
        )
        # sd_forward(prompt)
        print(f"Done, saved to {out_dir}")

    def edit_experiment(
        sd_model=None,
        src_image=None,
        out_dir: str = None,
        src_prompt: str = None,
        tgt_prompt: str = None,
        method: str = "sds",
    ):
        print("start editing experiment")

        latent = nn.Parameter(torch.empty(1, 4, 64, 64, device=device))
        # init with the image to be edited

        image_ = torchvision.transforms.functional.resize(src_image, (512, 512))
        image_ = torchvision.transforms.functional.to_tensor(image_)[:3, ...].unsqueeze(
            0
        )
        image_latent = sd_model.encode_imgs(image_.to(device))
        latent.data = image_latent.data
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        if method == "sds":
            edit_sds(
                sd_model, latent, tgt_prompt, f"{out_dir}/image_steps_sds.png", method
            )
        elif method == "nfsd":
            edit_nfsd(sd_model, latent, tgt_prompt, f"{out_dir}/image_steps_nfsd.png")
        elif method == "dds":
            edit_dds(
                sd_model,
                latent,
                src_prompt,
                tgt_prompt,
                f"{out_dir}/image_steps_dds.png",
            )
        elif method == "cpsd":
            edit_cpsd(
                sd_model,
                latent,
                src_prompt,
                tgt_prompt,
                f"{out_dir}/image_steps_cpsd.png",
            )

    args = parse_args()
    fix_randomness(42)
    out_dir = args.output_dir
    # init diffusion model
    device = torch.device("cuda")
    sd_model = StableDiffusion(device, version="2.1")
    if args.mode == "generation":
        out_dir = os.path.join(out_dir, "generation")
        tgt_prompt = "a zoomed out DSLR photo of a panda wearing a chef's hat and kneading bread dough on a countertop"
        generation_experiment(sd_model=sd_model, out_dir=out_dir, tgt_prompt=tgt_prompt)
    elif args.mode == "edit":
        out_dir = os.path.join(out_dir, "edit")

        src_prompt = "panda snowboarding"
        tgt_prompt = "squirrel snowboarding"

        with open("/root/autodl-tmp/CPSD/data/panda_snow.png", "rb") as f:
            image_ = Image.open(f)
            img_out_dir = os.path.join(
                out_dir,
                f"image_panda",
            )
            # for method in ["sds", "nfsd", "dds"]:
            for method in ["dds"]:
                edit_experiment(
                    sd_model=sd_model,
                    src_image=image_,
                    out_dir=img_out_dir,
                    src_prompt=src_prompt,
                    tgt_prompt=tgt_prompt,
                    method=method,
                )

    elif args.mode == "stylize":
        out_dir = os.path.join(out_dir, "edit")
        images_dir = "/root/autodl-tmp/CPSD/data/test_imgs"
        prompt_file = "/root/autodl-tmp/CPSD/data/test_imgs/prompt.txt"
        src_prompts = ["a photo of a horse"]
        with open(prompt_file, "r") as f:
            for prompt in f:
                src_prompts.append(prompt.strip())
        # tgt_prompts = [p + "in vincent van gogh style." for p in src_prompts]
        tgt_prompts = ["a fauvism painting of a horse on grassland."]

        i = 0
        # for all generated images
        for img_path in glob.glob(images_dir + "/*.png"):

            with open(img_path, "rb") as f:
                image_ = Image.open(f)
                img_out_dir = os.path.join(
                    out_dir,
                    f"image_{i}",
                )
                # edit with gt and stylized prompt
                src_prompt = src_prompts[i]
                tgt_prompt = tgt_prompts[i]
                edit_experiment(
                    sd_model=sd_model,
                    src_image=image_,
                    out_dir=img_out_dir,
                    src_prompt=src_prompt,
                    tgt_prompt=tgt_prompt,
                    method="cpsd",
                )
                i += 1
