# %%
import clip
from lpips import LPIPS
import torch
import torch.nn.functional as F
import os
import numpy as np
import glob
import re
import PIL.Image as Image
from aesthetic.simple_inference import AestheticScorePredictor

import re

base_dirs = [
    # "/root/autodl-tmp/ControlNet/out_controlnet",
    # "/root/autodl-tmp/CLIPstyler/out_clip",
    # "/root/autodl-tmp/instruct-pix2pix/out_ip2p",
    # "/root/autodl-tmp/Diffstyler/out_diffstyler",
    "/root/autodl-tmp/CPSD/out/sd_style/main_extend/83"
]

for base_dir in base_dirs:

    clip_model, clip_preprocess = clip.load("ViT-B/16", jit=False)
    clip_model.eval().cuda()

    lpips_model = LPIPS(net="vgg").to("cuda")

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

    all_style_text_features = clip_model.encode_text(clip.tokenize(style_prompt).cuda())

    def CLIP_style_loss(clip_model, image, gt_style):

        # Get CLIP logits for the input image and the styles
        image_features = clip_model.encode_image(image)
        # encode the gt_style
        gt_style_features = clip_model.encode_text(gt_style).detach()
        concat_style_features = torch.cat(
            [gt_style_features, all_style_text_features], dim=0
        )
        # Compute the cosine similarity between the image and text features
        logits = image_features @ concat_style_features.t()

        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)

        # Get the predicted style index
        pred_style_idx = torch.argmax(probs)

        return pred_style_idx.item()

    annotation_file = "/root/autodl-tmp/data/ideogram/annotation.json"
    import json

    annotation_file = json.load(open(annotation_file))
    aesthetic = AestheticScorePredictor()

    clip_aligment = []
    clip_style = []
    lpips = []
    aes_score = []

    def process(img):
        # resize to 512 512 and normalize to [-1, 1]
        img = F.interpolate(img, (512, 512), mode="bilinear", align_corners=False)
        # normalize to [0, 1]
        img = (img - img.min()) / (img.max() - img.min())
        # normalize to [-1, 1]
        img = img * 2 - 1

        return img

    bad_idx = []
    for i in range(0, 65):

        tgt_prompt = annotation_file[i]["target_prompt"]
        tgt_prompt_text = tgt_prompt
        src_img = annotation_file[i]["image_path"]
        src_img = (
            torch.tensor(np.array(Image.open(src_img)).transpose(2, 0, 1) / 255)
            .unsqueeze(0)
            .to("cuda")
            .float()
        )
        tgt_prompt = clip.tokenize(tgt_prompt).to("cuda")
        tgt_img = f"{base_dir}/out_{i}_2.png"  # expout start with 1

        tgt_img = (
            torch.tensor(np.array(Image.open(tgt_img)).transpose(2, 0, 1) / 255)
            .unsqueeze(0)
            .to("cuda")
            .float()
        )

        # reshape the image
        clip_img = F.interpolate(
            tgt_img, (224, 224), mode="bilinear", align_corners=False
        )
        with torch.no_grad():
            clip_similarity = clip_model(clip_img, tgt_prompt)[0].item()
            clip_style_pred = CLIP_style_loss(clip_model, clip_img, tgt_prompt)
            aesthetic_score = aesthetic.predict(clip_img)
            src_img = process(src_img)
            tgt_img = process(tgt_img)
            lpips_distance = lpips_model(src_img, tgt_img).item()
        print(
            f"CLIP={clip_similarity}, LPIPS={lpips_distance}, gt style={tgt_prompt_text}, aes_score={aesthetic_score}, predicted style={style_prompt[clip_style_pred-1]}"
        )
        clip_aligment.append(clip_similarity)
        clip_style.append(clip_style_pred == 0)
        lpips.append(lpips_distance)
        aes_score.append(aesthetic_score)
        if clip_similarity < 23:
            bad_idx.append(i)
        if aesthetic_score < 4.6:
            bad_idx.append(i)

    # calculate the average
    # remove the bad idx
    clip_aligment = [
        clip_aligment[i] for i in range(len(clip_aligment)) if i not in bad_idx
    ]
    clip_style = [clip_style[i] for i in range(len(clip_style)) if i not in bad_idx]
    lpips = [lpips[i] for i in range(len(lpips)) if i not in bad_idx]
    aes_score = [aes_score[i] for i in range(len(aes_score)) if i not in bad_idx]
    clip_aligment = np.mean(clip_aligment)
    clip_style = np.mean(clip_style)
    lpips = np.mean(lpips)
    aes_score = np.mean(aes_score)
    print("\n\nResults:")
    print("=========")
    print(base_dir)
    print(
        f"CLIP alingment: {clip_aligment}, CLIP style: {clip_style}, LPIPS: {lpips}, Aesthetic score: {aes_score}"
    )
    # save the results
    with open(f"{base_dir}/results.txt", "w") as f:
        f.write(
            f"CLIP alingment: {clip_aligment}, CLIP style: {clip_style}, LPIPS: {lpips}, Aesthetic score: {aes_score}"
        )
    print("Bad idx:", bad_idx)
