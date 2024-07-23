import os
import shutil

# Set the parent directory where the subfolders are located
parent_dir = "/root/autodl-tmp/CPSD/vis_out/explore"

# Loop through all the subdirectories
for subdir_name in os.listdir(parent_dir):
    subdir_path = os.path.join(parent_dir, subdir_name)
    if os.path.isdir(subdir_path):
        # Loop through all the files in the subdirectory
        for filename in os.listdir(subdir_path):
            # Check if the file is an image (you can modify the extensions as needed)
            if filename.endswith((".png", ".jpg", ".jpeg", ".gif")):
                src_path = os.path.join(subdir_path, filename)
                dst_path = os.path.join(subdir_path, subdir_name + "_" + filename)
                # Rename the file
                shutil.move(src_path, dst_path)
