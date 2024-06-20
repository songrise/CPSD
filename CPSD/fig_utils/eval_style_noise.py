import clip
from lpips_pytorch import LPIPS
import torch
import torch.nn.functional as F
import os
import numpy as np
import glob
import re
import PIL.Image as Image


base_dir = "/root/autodl-tmp/CPSD/out/ablation/stylized_noise"

clip_model = clip.load("ViT-B/16", jit=False)[0].eval().requires_grad_(False).to("cuda")
lpips_model = LPIPS(net_type="vgg").to("cuda")
mode = "calc"
if mode == "calc":
    lpips_dict = {
        "0": [],
        "10": [],
        # "15": [],
        "20": [],
        "25": [],
        "30": [],
        "35": [],
        "40": [],
        "45": [],
        "50": [],
    }
    clip_dict = {
        "0": [],
        "10": [],
        # "15": [],
        "20": [],
        "25": [],
        "30": [],
        "35": [],
        "40": [],
        "45": [],
        "50": [],
    }

    # for all subdir in base_dir
    for subdir in glob.glob(f"{base_dir}/*"):
        print(subdir)
        # if not pure number, skip
        if subdir.split("/")[-1] == "result":
            continue
        # for all files in subdir
        prompt_path = f"{subdir}/prompt.txt"
        with open(prompt_path, "r") as f:
            tgt_prompt = f.readlines()
        src_prompt = tgt_prompt[0].strip()
        tgt_prompt = tgt_prompt[1].strip()
        tgt_prompt = tgt_prompt[(len(src_prompt)) + 1 :].strip()
        src_image = f"{subdir}/stylenoise_0.png"
        # for all other images in subdir stylenoise_{}.png
        src_img = (
            torch.tensor(np.array(Image.open(src_image)).transpose(2, 0, 1) / 255)
            .unsqueeze(0)
            .to("cuda")
            .float()
        )
        for img_path in glob.glob(f"{subdir}/stylenoise_*.png"):
            # calculate CLIP and LPIPS distance
            img = (
                torch.tensor(np.array(Image.open(img_path)).transpose(2, 0, 1) / 255)
                .unsqueeze(0)
                .to("cuda")
                .float()
            )

            prompt = clip.tokenize(tgt_prompt).to("cuda")

            # reshape the image
            clip_img = F.interpolate(
                img, (224, 224), mode="bilinear", align_corners=False
            )
            with torch.no_grad():
                clip_similarity = clip_model(clip_img, prompt)[0].item()
                lpips_distance = lpips_model(src_img, img).item()
            print(f"{img_path}: CLIP={clip_similarity}, LPIPS={lpips_distance}")

            crt_step = img_path.split("_")[-1].split(".")[0]
            clip_dict[crt_step].append(clip_similarity)
            lpips_dict[crt_step].append(lpips_distance)
            # save the results
            # with open(f"{subdir}/results.txt", "a") as f:
            #     f.write(f"{img_path}: CLIP={clip_distance}, LPIPS={lpips_distance}\n")

    # dump the two dict to base dir
    import pickle

    with open(f"{base_dir}/result/clip.pkl", "wb") as f:
        pickle.dump(clip_dict, f)
    with open(f"{base_dir}/result/lpips.pkl", "wb") as f:
        pickle.dump(lpips_dict, f)
else:
    # %%
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np

    # load from pickle
    with open(
        "/root/autodl-tmp/CPSD/out/ablation/stylized_noise/result/clip.pkl", "rb"
    ) as f:
        clip_dict = pickle.load(f)
        # remove the "15" kv
        # del clip_dict["15"]
    with open(
        "/root/autodl-tmp/CPSD/out/ablation/stylized_noise/result/lpips.pkl", "rb"
    ) as f:
        lpips_dict = pickle.load(f)
        # del lpips_dict["15"]

    def remove_outliers_inplace(data, bottom_percentile, top_percentile):
        bottom_threshold = np.percentile(data, bottom_percentile)
        top_threshold = np.percentile(data, top_percentile)
        outlier_indices = [
            i
            for i, value in enumerate(data)
            if value < bottom_threshold or value > top_threshold
        ]
        for index in reversed(outlier_indices):
            del data[index]

    def normalize_data_inplace(data):
        min_value = min(data)
        max_value = max(data)
        for i in range(len(data)):
            data[i] = (data[i] - min_value) / (max_value - min_value)

    # Filter out outliers and normalize the data for LPIPS
    lpips_values = [value for sublist in lpips_dict.values() for value in sublist]
    remove_outliers_inplace(lpips_values, 1, 98)
    normalize_data_inplace(lpips_values)

    # Filter out outliers and normalize the data for CLIP
    clip_values = [value for sublist in clip_dict.values() for value in sublist]
    remove_outliers_inplace(clip_values, 0.1, 98)
    normalize_data_inplace(clip_values)

    # Update the dictionaries with the processed data
    lpips_dict_processed = {}
    clip_dict_processed = {}

    index = 0
    for key, values in lpips_dict.items():
        lpips_dict[key] = lpips_values[index : index + len(values)]
        index += len(values)

    index = 0
    for key, values in clip_dict.items():
        clip_dict[key] = clip_values[index : index + len(values)]
        index += len(values)

    # Calculate mean and standard deviation for each key
    lpips_mean = [np.mean(values) for values in lpips_dict.values()]
    lpips_std = [np.std(values) for values in lpips_dict.values()]
    clip_mean = [np.mean(values) for values in clip_dict.values()]
    clip_std = [np.std(values) for values in clip_dict.values()]

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    keys = [int(key) for key in lpips_dict.keys()]
    # Plot the mean curve and the shaded region for LPIPS
    ax.plot(keys, lpips_mean, label="LPIPS", color="blue", linewidth=2)
    ax.fill_between(
        keys,
        [m - s for m, s in zip(lpips_mean, lpips_std)],
        [m + s for m, s in zip(lpips_mean, lpips_std)],
        alpha=0.2,
        color="blue",
    )

    # Plot the mean curve and the shaded region for CLIP
    ax.plot(keys, clip_mean, label="CLIP", color="red", linewidth=2)
    ax.fill_between(
        keys,
        [m - s for m, s in zip(clip_mean, clip_std)],
        [m + s for m, s in zip(clip_mean, clip_std)],
        alpha=0.2,
        color="red",
    )

    # Set the title and labels
    ax.set_title("LPIPS and CLIP Metrics", fontsize=16)
    ax.set_xlabel("Key", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)

    # Set the legend
    ax.legend(fontsize=10)

    # Set the grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Set the tick font size
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Adjust the layout
    plt.tight_layout()

    # Display the plot
    plt.show()

# %%
