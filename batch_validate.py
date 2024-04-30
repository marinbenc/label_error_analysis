import subprocess
import sys

def run_validating():
    print("Using Python executable:", sys.executable)

    # Define the configurations for your models
    models = [
        "isic_unet_1e-1_8",
        "isic_unet_1e-2_8",
        "isic_unet_1e-3_8",
        "isic_unet_1e-4_8",
        "isic_unet_1e-5_8",
        "isic_unet_1e-6_8",
        "isic_unet_1e-1_16",
        "isic_unet_1e-2_16",
        "isic_unet_1e-3_16",
        "isic_unet_1e-4_16",
        "isic_unet_1e-5_16",
        "isic_unet_1e-6_16",
        "isic_unet_1e-1_32",
        "isic_unet_1e-2_32",
        "isic_unet_1e-3_32",
        "isic_unet_1e-4_32",
        "isic_unet_1e-5_32",
        "isic_unet_1e-6_32",
        "isic_unet_1e-1_64",
        "isic_unet_1e-2_64",
        "isic_unet_1e-3_64",
        "isic_unet_1e-4_64",
        "isic_unet_1e-5_64",
        "isic_unet_1e-6_64",
    ]

    # Define the dataset folder
    dataset_folder = "valid"

    for model in models:

        command = [sys.executable, "test.py", "seg_isic", model, "--dataset_folder", dataset_folder]
        print(f"Running: {' '.join(command)}")
        result = subprocess.run(command, text=True, capture_output=True, encoding='utf-8')
        
        if result.returncode == 0:
            print("Command executed successfully")
            print(result.stdout)
        else:
            print("Error in command execution")
            print(result.stderr)

if __name__ == '__main__':
    run_validating()