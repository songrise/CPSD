import os
import yaml
from itertools import combinations

output_folder = "/root/autodl-tmp/configs/CPSD"
os.makedirs(output_folder, exist_ok=True)

base_config = {
    "exp_name": "base_csdn",
    "batch_size": 1,
    "num_steps": 50,
    "out_path": "/root/autodl-tmp/CPSD/out/sd_style",
    "seed": 10,
    "share_resnet_layers": [0, 1],
    "share_attn": True,
    "share_query": False,
    "share_key": True,
    "share_value": False,
    "use_adain": True,
    "src_img": "/root/autodl-tmp/plug-and-play/data/horse.png",
    "src_prompt": "a photo of white horse on grassland",
    "style_cfg_scale": 7.5,
    "tau_attn": 1,
    "tau_feat": 1,
    "tgt_prompt": [
        "a photo of white horse on grassland",
        "a fauvism painting of a horse on grassland",
        "",
    ],
}

share_attn_layers_combinations = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [0, 1, 2, 3],
    [4, 5, 6, 7],
]

for idx, share_attn_layers in enumerate(share_attn_layers_combinations):
    config = base_config.copy()
    config["share_attn_layers"] = share_attn_layers
    yaml_path = os.path.join(output_folder, f"config_{idx}.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

print(
    f"Generated {len(share_attn_layers_combinations)} YAML configurations in {output_folder}"
)

# Generate the shell script
sh_script_path = os.path.join(output_folder, "run_experiments.sh")
with open(sh_script_path, "w") as f:
    f.write("#!/bin/bash\n\n")
    for idx in range(len(share_attn_layers_combinations)):
        config_path = os.path.join(output_folder, f"config_{idx}.yaml")
        cmd = f"python CSDN_inject.py --config {config_path} \n"
        f.write(cmd)

print(f"Generated shell script at {sh_script_path}")
