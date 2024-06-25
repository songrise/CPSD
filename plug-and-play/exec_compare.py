import os
import subprocess

# Define the base directories for the configurations
config_dir_extraction = "/root/autodl-tmp/plug-and-play/configs/my_compare"
config_dir_gen = "/root/autodl-tmp/plug-and-play/configs/my_compare_gen"

# Define the number of configurations to process
# Ns = [11, 16, 19]
Ns = range(4, 24)


# Function to run a command and print its output
def run_command(command):
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {command}")
        print(stderr.decode())
    else:
        print(stdout.decode())


# Step 1: Run feature extraction for each configuration
for i in Ns:
    config_path = os.path.join(config_dir_extraction, f"ideogram_{i}.yaml")
    command = f"python run_features_extraction.py --config {config_path}"
    print(f"Running: {command}")
    run_command(command)

    # # Step 2: Run PNP for each configuration
    # for i in Ns:
    config_path = os.path.join(config_dir_gen, f"ideogram_{i}.yaml")
    command = f"python run_pnp.py --config {config_path}"
    print(f"Running: {command}")
    run_command(command)

print("All commands executed.")
