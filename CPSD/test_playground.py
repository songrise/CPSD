# %%
import argparse, os

os.environ["HF_HOME"] = "/root/autodl-tmp/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/.cache"
from diffusers import DDIMScheduler, DiffusionPipeline
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == "__main__":

    pipe = DiffusionPipeline.from_pretrained(
        "playgroundai/playground-v2-512px-base",
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False,
        variant="fp16",
    )
    pipe.to("cuda")
    prompt = "A DSLR photo of an old man with a beard, wearing a hat and a coat, standing in the snow."
    image = pipe(prompt=prompt, width=512, height=512).images[0]
    image.show()
    # save
    image.save("/root/autodl-tmp/data/playground/test_out.png")
