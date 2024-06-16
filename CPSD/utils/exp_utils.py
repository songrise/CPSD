import uuid
import os
import PIL.Image as Image
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import torchvision


def make_unique_experiment_path(base_dir: str) -> str:
    """
    Create a unique directory in the base directory
    return: path to the unique directory
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    # Create a folder with a unique name for each experiment according to the process uid
    experiment_id = str(uuid.uuid4())[
        :8
    ]  # Generate a unique identifier for the experiment
    experiment_output_path = os.path.join(base_dir, experiment_id)
    os.makedirs(experiment_output_path)
    return experiment_output_path


def get_processed_image(image_dir: str, device) -> torch.Tensor:
    src_img = Image.open(image_dir)
    src_img = transforms.ToTensor()(src_img).unsqueeze(0).to(device)

    h, w = src_img.shape[-2:]
    src_img_512 = torchvision.transforms.functional.pad(
        src_img, ((512 - w) // 2,), fill=0, padding_mode="constant"
    )
    input_image = F.interpolate(
        src_img, (512, 512), mode="bilinear", align_corners=False
    )
    # drop alpha channel if it exists
    if input_image.shape[1] == 4:
        input_image = input_image[:, :3]

    return input_image


def seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    g_cpu = torch.Generator(device="cpu")
    g_cpu.manual_seed(42)
