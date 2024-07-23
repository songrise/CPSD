import os
import json

# Define the base directory for the configurations
base_dir = "/root/autodl-tmp/plug-and-play/configs/std"

# Ensure the directory exists
os.makedirs(base_dir, exist_ok=True)

# Define the number of configurations to generate
N = 4

# Template for the configuration file
config_template = """config:
  experiment_name: "ideogram_{N}"
  init_img: "/root/autodl-tmp/data/ideogram/{N}.png"
  ddim_steps: 50 # we use 999 steps for the best reconstruction
  save_feature_timesteps: 50
"""

# Generate and save the configuration files
for i in range(1, N + 1):
    config_content = config_template.format(N=i)
    config_filename = f"ideogram_{i}.yaml"
    config_path = os.path.join(base_dir, config_filename)

    with open(config_path, "w") as config_file:
        config_file.write(config_content)

print(f"Generated {N} configuration files in {base_dir}")


# Define the base directory for the configurations
base_dir = "/root/autodl-tmp/plug-and-play/configs/std_compare_gen"

# Ensure the directory exists
os.makedirs(base_dir, exist_ok=True)

# Define the number of configurations to generate


# Path to the annotation file
annotation_file = "/root/autodl-tmp/data/standard/annotation.json"

# Read the annotation file
with open(annotation_file, "r") as file:
    annotations = json.load(file)

# Template for the configuration file
config_template = """source_experiment_name: "ideogram_{N}"  # the experiment name of the source image
prompts: # text prompts for translations
  - "{src_prompt}"
  - "{tgt_prompt}"
#  - "a photo of a blue horse toy in playroom"
scale: 7.5 # unconditional guidance scale. Note that a higher value encourages deviation from the source image
num_ddim_sampling_steps: # if left empty, defaults to the ddim_steps arg used in the source experiment

# Control the level of structure preservation with injection timestep threshold
feature_injection_threshold: 80 # should be in [0, num_ddim_sampling_steps]

# Control the level of deviation from the source image with Negative prompting params.
negative_prompt: # if left blank, defaults to the source prompt
negative_prompt_alpha: 0.75 # ∈ [0, 1], determines the initial strength of negative-prompting (lower = stronger)
negative_prompt_schedule: "linear" 
# setting negative_prompt_alpha = 1.0, negative_prompt_schedule = "constant" is equivalent to not using negative prompting
"""

# Generate and save the configuration files
for i in range(1, N + 1):
    # Get the corresponding annotation
    annotation = annotations[i - 1]
    src_prompt = annotation["source_prompt"]
    tgt_prompt = annotation["target_prompt"]

    # Generate the configuration content
    config_content = config_template.format(
        N=i, src_prompt=src_prompt, tgt_prompt=tgt_prompt
    )

    # Define the filename and path for the configuration file
    config_filename = f"ideogram_{i}.yaml"
    config_path = os.path.join(base_dir, config_filename)

    # Write the configuration content to the file
    with open(config_path, "w") as config_file:
        config_file.write(config_content)

print(f"Generated {N} configuration files in {base_dir}")
