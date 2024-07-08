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


base_dir = "/root/autodl-tmp/CPSD/vis_out/ablate_delegation"

content_color = "#045985"  # Blue
style_color = "#EC3977"  # Pink
mode = "calc"
if mode == "calc":
    focus = "content"
    clip_model = (
        clip.load("ViT-B/16", jit=False)[0].eval().requires_grad_(False).to("cuda")
    )
    lpips_model = LPIPS(net_type="vgg").to("cuda")
    content_prompt = [
        "a cute fluffy cat with bright green eyes playing",
        "a tall glass skyscraper reaching high into the clouds",
        "crowded street with diverse people walking and shopping",
        "experienced lawyer with gray hair carrying a worn briefcase",
        "confident young woman in red dress walking city sidewalk",
        "delicious pizza with melted cheese and fresh colorful toppings",
        "serene mountain landscape with snowy peaks and clear lake",
        "colorful flower garden with roses, tulips, and exotic blooms",
        "busy city intersection with cars, pedestrians, and traffic lights",
        "cozy wooden cabin in forest clearing with smoking chimney",
        "majestic lion surveying territory from rock in African savanna",
        "peaceful beach with white sand, blue waves, and palms",
        "futuristic cityscape with holographic billboards and flying vehicles",
        "restored 1950s vintage automobile with chrome accents parked",
        "dense rainforest with towering trees, colorful birds, and waterfall",
        "charming small town with quaint shops and historic square",
        "vibrant farmers market with produce stands and artisanal crafts",
        "mysterious narrow alley between old buildings with flickering lamp",
        "grand medieval castle with towers, drawbridge, and stone walls",
        # "tranquil Japanese garden with koi pond and stone lanterns",
        # "group of friends enjoying picnic on checkered blanket",
        # "daring skydiver in freefall with colorful parachute opening",
        # "cozy bookshop interior with shelves, armchairs, and reading nooks",
        # "stunning underwater coral reef teeming with colorful tropical fish",
        # "bustling open-air market with vendors, produce, and handmade crafts",
        # "serene lakeside cabin surrounded by pine trees and mountains",
        # "historic European town square with fountain and old buildings",
        # "vibrant street art mural covering entire side of building",
        # "majestic snow-capped mountain peak glowing orange at sunrise",
        # "cozy coffee shop with steamy windows on rainy day",
        # "lively music festival scene with crowd and colorful stage",
        # "tranquil bamboo forest with sunlight filtering through leaves",
        # "colorful hot air balloons rising into clear blue sky",
        # "rustic red barn in countryside with rolling hills background",
        # "lively beach volleyball game with players diving for ball",
        # "serene Buddhist temple with golden statues and incense smoke",
        # "bustling city street at night with neon signs",
        # "cozy fireplace in log cabin with crackling flames",
        # "cute girl hugging fluffy pet dog in park",
        # "running cat chasing butterfly on sunny grass field",
        # "beautiful sunset over ocean with silhouetted palm trees",
        # "cozy living room with fireplace and comfortable armchairs",
        # "delicious pizza with melted cheese and crispy pepperoni slices",
        # "serene mountain landscape bathed in golden sunset light",
        # "colorful flower garden in full bloom with butterflies",
        # "busy city intersection at rush hour with cars",
        # "steaming cup of coffee on rustic wooden table",
        # "majestic lion with full mane resting in savanna",
        # "peaceful beach scene at sunrise with gentle waves",
        # "futuristic cityscape with flying cars and towering skyscrapers",
    ]

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

        # Get CLIP logits for the input image and the styles
        image_features = clip_model.encode_image(image)

        # Compute the cosine similarity between the image and text features
        logits = image_features @ all_style_text_features.T

        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)

        # Get the predicted style index
        pred_style_idx = torch.argmax(probs)
        pred_style = style_prompt[pred_style_idx]

        # get the predicted confidence for the ground truth style
        gt_index = style_prompt.index(gt_style)
        pred_confidence = probs[0, gt_index].item()

        # Check if the predicted style matches the ground truth
        if pred_style == gt_style:
            return 1, pred_confidence
        else:
            return 0, pred_confidence
            # get CLIP score

    lpips_dict = {
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

    clip_style_dict = {
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

    for injection_level in [-1]:
        for tau in ["5", "10", "15", "20", "25", "30", "35", "40", "45", "50"]:
            subdir = f"{base_dir}/tau_{tau}/{focus}_{injection_level}/"
            for imgs in glob.glob(f"{subdir}/*.png"):
                a, b, c = imgs.split("/")[-1].split("_")
                c = c.split(".")[0]
                if c == "style" or c == "content":
                    continue
                    # else c == final
                src_prompt = content_prompt[int(a)]
                tgt_prompt = style_prompt[int(b)]
                src_image = f"{subdir}/{a}_{b}_content.png"
                tgt_image = f"{subdir}/{a}_{b}_final.png"
                # for all other images in subdir stylenoise_{}.png
                src_img = (
                    torch.tensor(
                        np.array(Image.open(src_image)).transpose(2, 0, 1) / 255
                    )
                    .unsqueeze(0)
                    .to("cuda")
                    .float()
                )
                tgt_img = (
                    torch.tensor(
                        np.array(Image.open(tgt_image)).transpose(2, 0, 1) / 255
                    )
                    .unsqueeze(0)
                    .to("cuda")
                    .float()
                )

                prompt = clip.tokenize(tgt_prompt).to("cuda")

                # reshape the image
                clip_img = F.interpolate(
                    tgt_img, (224, 224), mode="bilinear", align_corners=False
                )
                with torch.no_grad():
                    clip_similarity = clip_model(clip_img, prompt)[0].item()
                    lpips_distance = lpips_model(src_img, tgt_img).item()
                    clip_style_pred, clip_style_confidence = CLIP_style_loss(
                        clip_model, clip_img, tgt_prompt
                    )
                print(
                    f"{subdir}: CLIP={clip_similarity}, LPIPS={lpips_distance}, CLIP_style={clip_style_pred}"
                )

                clip_dict[tau].append(clip_similarity)
                lpips_dict[tau].append(lpips_distance)
                clip_style_dict[tau].append(clip_style_pred)
                # clip_style_confidence_dict[crt_step].append(clip_style_confidence)
                # save the results
                # with open(f"{subdir}/results.txt", "a") as f:
                #     f.write(f"{img_path}: CLIP={clip_distance}, LPIPS={lpips_distance}\n")

            # dump the two dict to base dir
        import pickle

        with open(f"{base_dir}/{focus}_{injection_level}_clip.pkl", "wb") as f:
            pickle.dump(clip_dict, f)
        with open(f"{base_dir}/{focus}_{injection_level}_lpips.pkl", "wb") as f:
            pickle.dump(lpips_dict, f)
        with open(f"{base_dir}/{focus}_{injection_level}_clip_style.pkl", "wb") as f:
            pickle.dump(clip_style_dict, f)
else:
    # %%
    import torch
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np

    line_alpha = 0.7
    line_width = 3

    base_dir = "/root/autodl-tmp/CPSD/vis_out/ablate_delegation"
    content_idx = -1
    style_idx = content_idx
    # load from pickle
    focus = "content"

    # del clip_dict["15"]
    with open(f"{base_dir}/{focus}_{content_idx}_lpips.pkl", "rb") as f:
        lpips_dict = pickle.load(f)
        # del lpips_dict["15"]

    with open(
        f"{base_dir}/{focus}_{content_idx}_clip_style.pkl",
        "rb",
    ) as f:
        clip_dict = pickle.load(f)

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
    # remove_outliers_inplace(lpips_values, 0.1, 98)
    normalize_data_inplace(lpips_values)

    # calculate percentil of clip style
    clip_style_values = [sum(sublist) / len(sublist) for sublist in clip_dict.values()]

    normalize_data_inplace(clip_style_values)

    # Update the dictionaries with the processed data
    lpips_dict_processed = {}
    clip_processed = {}

    index = 0
    for key, values in lpips_dict.items():
        lpips_dict[key] = lpips_values[index : index + len(values)]
        index += len(values)

    index = 0
    for key, values in clip_dict.items():
        clip_dict[key] = clip_style_values[index : index + len(values)]
        index += len(values)

    # Calculate mean and standard deviation for each key
    lpips_mean = [np.mean(values) for values in lpips_dict.values()][::-1] + [0]
    lpips_std = [np.std(values) for values in lpips_dict.values()][::-1] + [0]
    # #!HARDCODED Jun 24: replace as style score just for
    clip_mean = clip_style_values[::-1] + [0]
    # clip_mean = [np.mean(values) for values in clip_dict.values()]
    # clip_std = [np.std(values) for values in clip_dict.values()]
    # clip_mean = [np.mean(values) for values in clip_dict.values()]
    # clip_std = [np.std(values) for values in clip_dict.values()]

    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    keys = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # Plot the mean curve and the shaded region for LPIPS
    ax.plot(
        keys,
        lpips_mean,
        label="Content Modification (controlled)",
        color=content_color,
        linewidth=line_width,
        marker="o",
        markersize=7,
        alpha=line_alpha,
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
        label="Style Strength (controlled)",
        color=style_color,
        linewidth=line_width,
        linestyle="-",
        marker="s",
        markersize=7,
        alpha=line_alpha,
    )

    if False:  # add theory lines
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

        def cum_score(low_t, hi_t, alphas, scale=1):
            res = 0
            for k in range(low_t + 1, hi_t):
                for j in range(low_t + 1, k - 1):
                    res += A_t(j, alphas) * B_t(k, alphas) * scale
            return res

        content_scores = []
        style_scores = []

        for tau in range(0, 50):
            crt_content = cum_score(50 - tau, 50, sampled_alphas, scale=1)
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
            color=content_color,
            linewidth=3.5,
            linestyle=":",
            marker="P",
            markersize=7,
            alpha=line_alpha,
        )
        ax.plot(
            keys,
            style_scores,
            label="Style Strength (Theory)",
            color=style_color,
            linewidth=3.5,
            linestyle=":",
            marker="X",
            markersize=7,
            alpha=line_alpha,
        )
    if focus == "content":  # plot upper bound
        uncontrolled_content = [
            0.8436280071550781,
            0.7432659436886719,
            0.6082894277257476,
            0.46761277699729736,
            0.3442963862894525,
            0.23845040554129743,
            0.15020915887478334,
            0.08333844913144473,
            0.03779865027634021,
            0.011826624269346407,
            0,
        ]
        uncontrolled_style = [
            1.0,
            0.7934272300469482,
            0.572769953051643,
            0.35680751173708913,
            0.21126760563380279,
            0.1173708920187793,
            0.051643192488262886,
            0.03286384976525821,
            0.023474178403755853,
            0.014084507042253499,
            0,
        ]
        ax.plot(
            keys,
            uncontrolled_content,
            label="Content Modification (w/o content control)",
            color=content_color,
            linewidth=line_width,
            linestyle="--",
            marker="P",
            markersize=7,
            alpha=line_alpha,
        )
        ax.plot(
            keys,
            uncontrolled_style,
            label="Style Strength (w/o content control)",
            color=style_color,
            linewidth=line_width,
            linestyle="--",
            marker="X",
            markersize=7,
            alpha=line_alpha,
        )
    if focus == "style" and True:  # plot lower bound
        uncontrolled_content = [
            0.6297541249795648,
            0.41919996562184264,
            0.3337598163680474,
            0.2646224593932166,
            0.20097911527499307,
            0.14156115910263273,
            0.09577791680155585,
            0.06045422406222578,
            0.032316110937234035,
            0.009408131074687905,
            0,
        ]

        uncontrolled_content = [
            0.5250192779682478,
            0.37607702451741487,
            0.3062213401576595,
            0.24516020248418222,
            0.18783412219695517,
            0.13630733925238403,
            0.09301406679419189,
            0.057561558457034354,
            0.029772007667804457,
            0.008419782831535244,
            0,
        ]
        uncontrolled_style = [0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ax.plot(
            keys,
            uncontrolled_content,
            label="Content Modification (w/o style control)",
            color=content_color,
            linewidth=line_width,
            linestyle="--",
            marker="P",
            markersize=7,
            alpha=line_alpha,
        )
        ax.plot(
            keys,
            uncontrolled_style,
            label="Style Strength (w/o style control)",
            color=style_color,
            linewidth=line_width,
            linestyle="--",
            marker="X",
            markersize=7,
            alpha=line_alpha,
        )
    # Set the title and labels
    ax.set_title(r"b) Relative Content and Style Over Time", fontsize=14)
    ax.set_xlabel(r"Stylization Start Step $\tau$", fontsize=14)
    ax.set_ylabel("Relative Value", fontsize=14)

    # # Set the x-ticks positions
    xticks = [0, 10, 20, 30, 40, 50]
    ax.set_xticks(xticks)

    # # Set the x-tick labels in reversed order
    xticklabels = [str(tick) for tick in xticks][::-1]
    xticklabels = ["T", "0.8T", "0.6T", "0.4T", "0.2T", "0"]
    ax.set_xticklabels(xticklabels)

    # Set the legend
    ax.legend(fontsize=10)

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
    fig.savefig(f"{base_dir}/fig_traj_{focus}_{content_idx}.png", dpi=300)
    print(f"Plot saved to {base_dir}/fig_traj_{focus}_{content_idx}.png")
# %%
