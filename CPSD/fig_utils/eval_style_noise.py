# %%
import clip
from lpips_pytorch import LPIPS
import torch
import torch.nn.functional as F
import os
import numpy as np
import glob
import re
import PIL.Image as Image

import re


base_dir = "/root/autodl-tmp/CPSD/out/ablation/stylized_noise_2"

clip_model = clip.load("ViT-B/16", jit=False)[0].eval().requires_grad_(False).to("cuda")
lpips_model = LPIPS(net_type="vgg").to("cuda")
mode = "calac"
if mode == "calc":
    all_styles = [
        "watercolor style",
        "fauvism style",
        "pencil sketch style",
        "pointillism style",
        "art deco style",
        "impressionism style",
        "surrealism style",
        "pop art style",
        "cubism style",
        "abstract expressionism style",
    ]
    all_style_text_features = clip_model.encode_text(clip.tokenize(all_styles).cuda())

    def CLIP_style_loss(clip_model, image, gt_style):
        """
        Return 1 if the prediction matches the ground truth style, 0 otherwise.

        Args:
            clip_model (CLIP): The CLIP model instance.
            image (torch.Tensor): The input image tensor.
            gt_style (str): The ground truth style in the format "xxx x x in xxx style".

        Returns:
            int: 1 if the prediction matches the ground truth style, 0 otherwise.
            float: predicted confidence
        """

        parts = re.split(r"\b(in)\b", gt_style)
        gt_style_extracted = parts[2].strip()
        # Get CLIP logits for the input image and the styles
        image_features = clip_model.encode_image(image)

        # Compute the cosine similarity between the image and text features
        logits = image_features @ all_style_text_features.T

        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)

        # Get the predicted style index
        pred_style_idx = torch.argmax(probs)
        pred_style = all_styles[pred_style_idx]

        # get the predicted confidence for the ground truth style
        gt_index = all_styles.index(gt_style_extracted)
        pred_confidence = probs[0, gt_index].item()

        # Check if the predicted style matches the ground truth
        if pred_style == gt_style_extracted:
            return 1, pred_confidence
        else:
            return 0, pred_confidence
            # get CLIP score

    lpips_dict = {
        "0": [],
        "5": [],
        "10": [],
        "15": [],
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
        "5": [],
        "10": [],
        "15": [],
        "20": [],
        "25": [],
        "30": [],
        "35": [],
        "40": [],
        "45": [],
        "50": [],
    }

    clip_style_conf_dict = {
        "0": [],
        "5": [],
        "10": [],
        "15": [],
        "20": [],
        "25": [],
        "30": [],
        "35": [],
        "40": [],
        "45": [],
        "50": [],
    }

    clip_style_confidence_dict = {
        "0": [],
        "5": [],
        "10": [],
        "15": [],
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
                clip_style_pred, clip_style_confidence = CLIP_style_loss(
                    clip_model, clip_img, tgt_prompt
                )
            print(
                f"{img_path}: CLIP={clip_similarity}, LPIPS={lpips_distance}, CLIP_style={clip_style_pred}, confidence={clip_style_confidence}"
            )

            crt_step = img_path.split("_")[-1].split(".")[0]
            clip_dict[crt_step].append(clip_similarity)
            lpips_dict[crt_step].append(lpips_distance)
            clip_style_conf_dict[crt_step].append(clip_style_pred)
            clip_style_confidence_dict[crt_step].append(clip_style_confidence)
            # save the results
            # with open(f"{subdir}/results.txt", "a") as f:
            #     f.write(f"{img_path}: CLIP={clip_distance}, LPIPS={lpips_distance}\n")

    # dump the two dict to base dir
    import pickle

    with open(f"{base_dir}/result/clip.pkl", "wb") as f:
        pickle.dump(clip_dict, f)
    with open(f"{base_dir}/result/lpips.pkl", "wb") as f:
        pickle.dump(lpips_dict, f)
    with open(f"{base_dir}/result/clip_style.pkl", "wb") as f:
        pickle.dump(clip_style_conf_dict, f)
    with open(f"{base_dir}/result/clip_style_confidence.pkl", "wb") as f:
        pickle.dump(clip_style_confidence_dict, f)
else:
    # %%
    import torch
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np

    base_dir = "/root/autodl-tmp/CPSD/out/ablation/stylized_noise_2"
    # load from pickle
    with open(f"{base_dir}/result/clip.pkl", "rb") as f:
        clip_dict = pickle.load(f)
        # remove the "15" kv
        # del clip_dict["15"]
    with open(f"{base_dir}/result/lpips.pkl", "rb") as f:
        lpips_dict = pickle.load(f)
        # del lpips_dict["15"]

    with open(f"{base_dir}/result/clip.pkl", "rb") as f:
        clip_dict = pickle.load(f)

    with open(
        f"{base_dir}/result/clip_style.pkl",
        "rb",
    ) as f:
        clip_style_conf_dict = pickle.load(f)
    with open(
        f"{base_dir}/result/clip_style_confidence.pkl",
        "rb",
    ) as f:
        clip_style_confidence_dict = pickle.load(f)

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
    remove_outliers_inplace(lpips_values, 0.1, 98)
    normalize_data_inplace(lpips_values)

    # Filter out outliers and normalize the data for CLIP
    clip_values = [value for sublist in clip_dict.values() for value in sublist]
    remove_outliers_inplace(clip_values, 0.1, 98)
    normalize_data_inplace(clip_values)

    # calculate percentil of clip style
    clip_style_values = [
        sum(sublist) / len(sublist) for sublist in clip_style_conf_dict.values()
    ]

    normalize_data_inplace(clip_style_values)

    clip_style_confidence_values = [
        value for sublist in clip_style_confidence_dict.values() for value in sublist
    ]
    remove_outliers_inplace(clip_style_confidence_values, 0.1, 98)
    normalize_data_inplace(clip_style_confidence_values)

    # Update the dictionaries with the processed data
    lpips_dict_processed = {}
    clip_dict_processed = {}
    clip_style_processed = {}
    clip_style_confidence_processed = {}

    index = 0
    for key, values in lpips_dict.items():
        lpips_dict[key] = lpips_values[index : index + len(values)]
        index += len(values)

    index = 0
    for key, values in clip_dict.items():
        clip_dict[key] = clip_values[index : index + len(values)]
        index += len(values)

    index = 0
    for key, values in clip_style_conf_dict.items():
        clip_style_conf_dict[key] = clip_style_confidence_values[
            index : index + len(values)
        ]
        index += len(values)

    # Calculate mean and standard deviation for each key
    lpips_mean = [np.mean(values) for values in lpips_dict.values()][::-1]
    # normalize the mean
    normalize_data_inplace(lpips_mean)

    lpips_std = [np.std(values) for values in lpips_dict.values()][::-1]
    # #!HARDCODED Jun 24: replace as style score just for
    clip_mean = clip_style_values[::-1]
    # clip_mean = [np.mean(values) for values in clip_dict.values()]
    # clip_std = [np.std(values) for values in clip_dict.values()]
    # clip_mean = [np.mean(values) for values in clip_style_conf_dict.values()]
    # clip_std = [np.std(values) for values in clip_style_conf_dict.values()]

    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    keys = [int(key) for key in lpips_dict.keys()]
    # Plot the mean curve and the shaded region for LPIPS
    ax.plot(
        keys,
        lpips_mean,
        label="Content Modification (LPIPS)",
        color="#046995",
        linewidth=3,
        marker="o",
        markersize=7,
    )
    # ax.fill_between(
    #     keys,
    #     [m - s for m, s in zip(lpips_mean, lpips_std)],
    #     [m + s for m, s in zip(lpips_mean, lpips_std)],
    #     alpha=0.2,
    #     color="#46AEA0",
    # )

    # Plot the mean curve and the shaded region for CLIP
    ax.plot(
        keys,
        clip_mean,
        label="Style Strength (CLIP Style)",
        color="#EC3977",
        linewidth=3,
        linestyle="-",
        marker="s",
        markersize=7,
    )

    if True:  # add theory lines
        tensor_path = "/root/autodl-tmp/CPSD/vis_out/traj/alpha_cum.pt"
        alphas = torch.load(tensor_path)

        # Extract samples every 20 steps, assuming there are sufficient elements.
        stride = 20
        num_samples = 50
        indices = [i * stride for i in range(num_samples)]

        # Ensure indices are within the bounds of the tensor size
        if indices[-1] >= len(alphas):
            raise ValueError(
                "The tensor does not have enough elements to sample 50 values with the given stride."
            )

        # Sample the tensor using the calculated indices
        sampled_alphas = alphas[indices]

        # Convert tensor to numpy array for plotting
        sampled_alphas_np = sampled_alphas.numpy()

        def A_t(t, alphas):

            return np.sqrt(alphas[t - 1]) / np.sqrt(alphas[t])

        def B_t(t, alphas):

            return np.sqrt(1 - alphas[t - 1]) - np.sqrt(
                alphas[t] * (1 - alphas[(t - 1)])
            )

        def cum_score(low_t, hi_t, alphas):
            res = 0
            for k in range(low_t + 1, hi_t):
                for j in range(low_t + 1, k - 1):
                    res += A_t(j, alphas) * B_t(k, alphas)
            return res

        content_scores = []
        style_scores = []

        for tau in range(0, 50):
            crt_content = cum_score(50 - tau, 50, sampled_alphas)
            crt_style = cum_score(0, tau, sampled_alphas)
            content_scores.append(crt_content)
            style_scores.append(crt_style)
        content_scores = np.array(content_scores)
        style_scores = np.array(style_scores)
        normalize_data_inplace(content_scores)
        normalize_data_inplace(style_scores)
        # select the [0, 10, 20, 25, 30, 35, 40, 45, 50] points
        # keys = [0, 9, 19, 24, 29, 34, 39, 44, 49]
        keys = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
        content_scores = content_scores[keys][::-1]
        style_scores = style_scores[keys][::-1]

        ax.plot(
            keys,
            content_scores,
            label="Content Modification (Theory)",
            color="#046995",
            linewidth=3.5,
            linestyle=":",
            marker="P",
            markersize=7,
        )
        ax.plot(
            keys,
            style_scores,
            label="Style Strength (Theory)",
            color="#EC3977",
            linewidth=3.5,
            linestyle=":",
            marker="X",
            markersize=7,
        )

    # Set the title and labels
    # ax.set_title(r"b) Relative Content and Style Over Time", fontsize=14)
    ax.set_xlabel(r"Stylization Start Step $\tau$", fontsize=14)
    ax.set_ylabel("Normalized Value", fontsize=14)

    # Set the x-ticks positions
    xticks = [0, 10, 20, 30, 40, 50]
    ax.set_xticks(xticks)

    # Set the x-tick labels in reversed order
    xticklabels = [str(tick) for tick in xticks][::-1]
    xticklabels = ["T", "0.8T", "0.6T", "0.4T", "0.2T", "0"]
    ax.set_xticklabels(xticklabels)

    # Set the legend
    ax.legend(fontsize=12)

    # Set the grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Set the tick font size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Adjust the layout
    plt.tight_layout()

    # Display the plot
    plt.show()
    # save the plot
    fig.savefig(f"{base_dir}/result/fig_traj_b.png", dpi=300)
    print(f"Plot saved to {base_dir}/result/fig_traj_b.png")
# %%
