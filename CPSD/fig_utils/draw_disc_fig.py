import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import clip
from lpips_pytorch import LPIPS
import torch
import torch.nn.functional as F
import os
import numpy as np
import glob
import re
import PIL.Image as Image
from matplotlib import colors
import re


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
    "tranquil Japanese garden with koi pond and stone lanterns",
    "group of friends enjoying picnic on checkered blanket",
    "daring skydiver in freefall with colorful parachute opening",
    "cozy bookshop interior with shelves, armchairs, and reading nooks",
    "stunning underwater coral reef teeming with colorful tropical fish",
    "bustling open-air market with vendors, produce, and handmade crafts",
    "serene lakeside cabin surrounded by pine trees and mountains",
    "historic European town square with fountain and old buildings",
    "vibrant street art mural covering entire side of building",
    "majestic snow-capped mountain peak glowing orange at sunrise",
    "cozy coffee shop with steamy windows on rainy day",
    "lively music festival scene with crowd and colorful stage",
    "tranquil bamboo forest with sunlight filtering through leaves",
    "colorful hot air balloons rising into clear blue sky",
    "rustic red barn in countryside with rolling hills background",
    "lively beach volleyball game with players diving for ball",
    "serene Buddhist temple with golden statues and incense smoke",
    "bustling city street at night with neon signs",
    "cozy fireplace in log cabin with crackling flames",
    "cute girl hugging fluffy pet dog in park",
    "running cat chasing butterfly on sunny grass field",
    "beautiful sunset over ocean with silhouetted palm trees",
    "cozy living room with fireplace and comfortable armchairs",
    "delicious pizza with melted cheese and crispy pepperoni slices",
    "serene mountain landscape bathed in golden sunset light",
    "colorful flower garden in full bloom with butterflies",
    "busy city intersection at rush hour with cars",
    "steaming cup of coffee on rustic wooden table",
    "majestic lion with full mane resting in savanna",
    "peaceful beach scene at sunrise with gentle waves",
    "futuristic cityscape with flying cars and towering skyscrapers",
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

mode = "calc"
if mode == "calc":

    base_dir = "/root/autodl-tmp/CPSD/vis_out/control_pnp"

    clip_model = (
        clip.load("ViT-B/16", jit=False)[0].eval().requires_grad_(False).to("cuda")
    )
    lpips_model = LPIPS(net_type="vgg").to("cuda")

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

    result_tensor = torch.zeros(4, 4, 2)
    result_std_tensor = torch.zeros(4, 4, 2)

    # for all subdir in base_dir
    for c_i in range(0, 4):
        for s_i in range(0, 4):
            c_s_result_dir = f"{base_dir}/content_{c_i}_style_{s_i}"
            crt_lpips = []
            crt_clip = []
            for imgs in glob.glob(f"{c_s_result_dir}/*.png"):
                a, b, c = imgs.split("/")[-1].split("_")
                c = c.split(".")[0]
                if c == "style" or c == "content":
                    continue
                # else c == final
                src_prompt = content_prompt[int(a)]
                tgt_prompt = style_prompt[int(b)]

                src_image = f"{c_s_result_dir}/{a}_{b}_content.png"
                tgt_image = f"{c_s_result_dir}/{a}_{b}_final.png"
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
                    f"{tgt_image}: CLIP={clip_similarity}, LPIPS={lpips_distance}, CLIP_style={clip_style_pred}"
                )

                crt_lpips.append(lpips_distance)
                crt_clip.append(clip_style_pred)
            result_tensor[c_i, s_i, 0] = np.mean(crt_lpips)
            result_tensor[c_i, s_i, 1] = np.mean(crt_clip)
            result_std_tensor[c_i, s_i, 0] = np.std(crt_lpips)
            result_std_tensor[c_i, s_i, 1] = np.std(crt_clip)

        # save the result tensor
        torch.save(result_tensor, f"{base_dir}/result_tensor.pt")
        torch.save(result_std_tensor, f"{base_dir}/result_std_tensor.pt")
else:
    exp_name = "control_pnp"
    # Initialize the data
    H, W = 4, 4
    # data = np.random.rand(H, W, 2)  # Assuming random values for demonstration
    result_path = f"/root/autodl-tmp/CPSD/vis_out/{exp_name}/result_tensor.pt"
    # result_std_tensor_path = (
    #     f"/root/autodl-tmp/CPSD/vis_out/{exp_name}/result_std_tensor.pt"
    # )
    data = torch.load(result_path)
    # data_std = torch.load(result_std_tensor_path)
    # Parameters for the disc sizes
    min_radius = 0.1  # Corresponds to 50% of the maximum radius
    max_radius = 0.35  # Corresponds to 100% of the maximum radius

    #####processing
    lpips_score, clip_style = data[:, :, 0], data[:, :, 1]
    # Normalize the data
    lpips_score = (lpips_score - lpips_score.min()) / (
        lpips_score.max() - lpips_score.min()
    )

    # normalize the clip style at each content level
    # clip_style_per_content = [clip_style[i, :] for i in range(4)]
    # for i in range(4):
    #     clip_style_per_content[i] = (
    #         clip_style_per_content[i] - clip_style_per_content[i].min()
    #     ) / (clip_style_per_content[i].max() - clip_style_per_content[i].min())
    #     clip_style = torch.stack(clip_style_per_content)

    # normalize as a whole
    clip_style_global_normalized = (clip_style - clip_style.min()) / (
        clip_style.max() - clip_style.min()
    )
    # then, scale each row of the global-normalized value by a linear mapping
    # because the difference beween min and max inside each row can hardly be visualized in bubble plot
    if exp_name == "control_pnp":
        for i in range(4):
            row_min, row_max = clip_style[i].min(), clip_style[i].max()
            # if row_max - row_min < 0.1 and i != 3:
            if i == 3:
                # mapping [row_min, row_max] to [0, row_max]
                # clip_style[i] = (
                #     (clip_style[i] - row_min) * row_max / (row_max - row_min)
                # )
                pass
            elif i < 3:
                next_row_min, next_row_max = (
                    clip_style[i + 1].min(),
                    clip_style[i + 1].max(),
                )
                offseted_min = min(row_min, next_row_min)
                # mapping [row_min, row_max] to [next_row_min,row_max]
                clip_style[i] = (clip_style[i] - row_min) * (row_max - offseted_min) / (
                    row_max - row_min
                ) + offseted_min

    else:
        for i in range(4):
            row_min, row_max = clip_style[i].min(), clip_style[i].max()
            # if row_max - row_min < 0.1 and i != 3:
            if i == 3:
                prev_row_min, prev_row_max = (
                    clip_style[i - 1].min(),
                    clip_style[i - 1].max(),
                )
                # mapping [row_min, row_max] to [0, prev_row_min]
                offseted_max = max(row_max, prev_row_min) + 0.07
                clip_style[i] = (
                    (clip_style[i] - row_min) * offseted_max / (row_max - row_min)
                )
            elif i < 3:
                next_row_min, next_row_max = (
                    clip_style[i + 1].min(),
                    clip_style[i + 1].max(),
                )
                if i == 0:
                    offseted_min = min(row_min, next_row_min) - 0.07
                else:
                    offseted_min = min(row_min, next_row_min) - 0.15
                # mapping [row_min, row_max] to [next_row_min,row_max]
                clip_style[i] = (clip_style[i] - row_min) * (row_max - offseted_min) / (
                    row_max - row_min
                ) + offseted_min

    # apply a monotonic, non-liearn mapping for the lpips (power of 2) to make easier to distinguish
    data[:, :, 0] = lpips_score**2
    data[:, :, 1] = clip_style**1
    data = torch.flip(data, [0])

    def plot_discs(data, min_radius, max_radius, offset, fig_width, fig_height):
        # Define the colors
        content_color = "#045985"  # Blue
        style_color = "#EC3977"  # Pink
        H, W, _ = data.shape

        # Create the figure and axis with specified size
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Set limits with some padding
        ax.set_xlim(-0.5, W - 0.2)
        ax.set_ylim(-0.5, H - 0.2)

        # Create chessboard pattern
        chessboard = np.zeros((H, W))
        chessboard[::2, ::2] = 1
        chessboard[1::2, 1::2] = 1

        extent = (-0.5, W - 0.5, -0.5, H - 0.5)
        ax.imshow(chessboard, cmap="gray", alpha=0.1, extent=extent, aspect="equal")

        for i in range(H):
            for j in range(W):
                # Get the values for the discs
                value1 = data[i, j, 0]
                value2 = data[i, j, 1]

                # Calculate the diameters
                diameter1 = 2 * (min_radius + value1 * (max_radius - min_radius))
                diameter2 = 2 * (min_radius + value2 * (max_radius - min_radius))

                # Calculate the offsets
                offset_value = offset / 2.0
                x_pos1 = j - offset_value
                x_pos2 = j + offset_value

                # Plot the ellipses
                ellipse1 = Ellipse(
                    (x_pos1, H - i - 1),
                    width=diameter1,
                    height=diameter1,
                    color=content_color,
                    alpha=0.5,
                )
                ellipse2 = Ellipse(
                    (x_pos2, H - i - 1),
                    width=diameter2,
                    height=diameter2,
                    color=style_color,
                    alpha=0.5,
                )
                ax.add_artist(ellipse1)
                ax.add_artist(ellipse2)

                # Plot projections on top edge
                projection_height = 0.13
                ax.add_patch(
                    Rectangle(
                        (j - diameter1 / 2 - +offset_value, H - 0.46),
                        diameter1,
                        projection_height,
                        facecolor=content_color,
                        alpha=0.33,
                    )
                )
                ax.add_patch(
                    Rectangle(
                        (
                            j - diameter2 / 2 + offset_value,
                            H - 0.46 + projection_height,
                        ),
                        diameter2,
                        projection_height,
                        facecolor=style_color,
                        alpha=0.33,
                    )
                )

                # Plot projections on right edge
                projection_width = 0.13
                ax.add_patch(
                    Rectangle(
                        (W - 0.46 + projection_width, H - i - 1 - diameter1 / 2),
                        projection_width,
                        diameter1,
                        facecolor=content_color,
                        alpha=0.33,
                    )
                )
                ax.add_patch(
                    Rectangle(
                        (W - 0.46, H - i - 1 - diameter2 / 2),
                        projection_width,
                        diameter2,
                        facecolor=style_color,
                        alpha=0.33,
                    )
                )

        # Add vertical dotted lines
        for j in range(W):
            offset_value = offset / 2.0
            # Left line of the pair
            ax.axvline(
                x=j - offset_value,
                color="gray",
                linestyle=":",
                alpha=0.2,
                zorder=0,
                lw=2,
            )
            # Right line of the pair
            ax.axvline(
                x=j + offset_value,
                color="gray",
                linestyle=":",
                alpha=0.2,
                zorder=0,
                lw=2,
            )
            # Add horizontal dotted lines
        for i in range(H):
            ax.axhline(y=i, color="gray", linestyle=":", alpha=0.2, zorder=0, lw=2)

        # Set the ticks and labels for the grid
        ax.set_xticks(range(W))
        ax.set_yticks(range(H))
        x_ticks = ["0", "4", r"4$\to$8", r"4$\to$12"]
        y_ticks = ["0", "4", r"4$\to$8", r"4$\to$12"]
        ax.set_xticklabels(x_ticks, fontsize=14)
        ax.set_yticklabels(
            y_ticks, rotation=90, va="center", fontsize=14
        )  # Rotate labels 90 degrees
        # ax.set_xlabel("Style control layers", fontsize=14)
        # ax.set_ylabel("Content control layers", fontsize=14)

        # Ensure the aspect of the plot is equal
        ax.set_aspect("equal", adjustable="box")

        # Create a title with colored words
        # title = ax.set_title("Relative Content and Style Control", fontsize=16)

    # Plot the discs with an offset and custom figure size
    offset = 0.3  # Adjust the offset as needed
    fig_width, fig_height = 7, 6  # Adjust these values to set the figure size
    plot_discs(data, min_radius, max_radius, offset, fig_width, fig_height)
    plt.tight_layout(pad=0.1)
    # save the plot
    plt.savefig(
        f"/root/autodl-tmp/CPSD/vis_out/{exp_name}/result_discs_{exp_name}.png", dpi=300
    )
    print(
        f"Saved the plot to /root/autodl-tmp/CPSD/vis_out/{exp_name}/result_discs_{exp_name}.png"
    )
    # %%
