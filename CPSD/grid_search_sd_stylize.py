import os
import itertools
import yaml

# Define the parameter ranges for the grid search
tau_attn_range = [5, 10]
tau_feat_range = [5, 10]
share_attn_subsets = [[0], list(range(5)), list(range(4, 9)), list(range(9))]
share_resnet_subsets = [[0], list(range(5)), list(range(4, 9)), list(range(9))]

# Define the base configuration
base_config = {
    "src_img": "/root/autodl-tmp/plug-and-play/data/horse.png",
    "out_path": "/root/autodl-tmp/CPSD/out/sd_style",
    "num_steps": 50,
    "seed": 10,
    "style_cfg_scale": 7.5,
    "batch_size": 1,
    "src_prompt": "A white horse on a green field, photorealistic style.",
    "tgt_prompt": [
        "A white horse on a green field, pencil sketch style.",
        "A white horse on a green field, abstract style.",
        "A white horse on a green field, impressionist style.",
        "A white horse on a green field, cubist style.",
        "A white horse on a green field, Vincent van gogh starry night style.",
    ],
}

# Create the output folder for YAML files
output_folder = "/root/autodl-tmp/configs/CPSD"
os.makedirs(output_folder, exist_ok=True)

# Generate the YAML files for each combination of parameters
config_files = []
for tau_attn, tau_feat, share_attn, share_resnet in itertools.product(
    tau_attn_range, tau_feat_range, share_attn_subsets, share_resnet_subsets
):
    # Create a copy of the base configuration
    config = base_config.copy()

    # Update the configuration with the current parameter values
    config["tau_attn"] = tau_attn
    config["tau_feat"] = tau_feat
    config["share_attn_layers"] = share_attn
    config["share_resnet_layers"] = share_resnet

    # Generate a unique filename for the YAML file
    filename = f"config_tau_attn_{tau_attn}_tau_feat_{tau_feat}_share_attn_{share_attn}_share_resnet_{share_resnet}.yaml"
    file_path = os.path.join(output_folder, filename)

    # Write the configuration to the YAML file
    with open(file_path, "w") as file:
        yaml.dump(config, file)

    config_files.append(file_path)

# Generate the shell script to execute the configurations
shell_script = "/root/autodl-tmp/configs/CPSD/run_configs.sh"
with open(shell_script, "w") as file:
    file.write("#!/bin/bash\n\n")
    for config_file in config_files:
        file.write(f'nohup python sd_stylize.py --config "{config_file}"\n')

# Make the shell script executable
os.chmod(shell_script, 0o755)
