# Define the different configurations for learning rates and batch sizes

import subprocess
import sys
import os

def run_training():
    print("Using Python executable:", sys.executable)

    # Define the different configurations for learning rates and batch sizes
    learning_rates = ["1e-1", "1e-2", "1e-3", "1e-4", "1e-5", "1e-6"]]
    batch_sizes = [8, 16, 32, 64]

    # Define the constant parameters
    dataset = "seg_isic"
    overwrite = "--overwrite"
    workers = "--workers 0"

    # Iterate over each combination of learning rate and batch size
    for lr in learning_rates:
        for batch_size in batch_sizes:
            log_name = f"isic_unet_{lr}_{batch_size}"
            command = [
                sys.executable, "train.py",
                "--dataset", dataset,
                "--log_name", log_name,
                overwrite,
                "--lr", lr,
                "--batch-size", str(batch_size),
                workers
            ]

            print(f"Running: {' '.join(command)}")
            env = os.environ.copy()
            result = subprocess.run(command, text=True, capture_output=True, encoding='utf-8', shell=True, env=env)

            if result.returncode == 0:
                print("Command executed successfully")
                print(result.stdout)
            else:
                print("Error in command execution")
                print(result.stderr)
            # print(f"Running: {' '.join(command)}")
            # try:
            #     result = subprocess.run(command, text=True, capture_output=True, encoding='utf-8', shell=True)
            #     if result.returncode == 0:
            #         print("Command executed successfully")
            #         print(result.stdout)
            #     else:
            #         raise Exception(f"Error in command execution\n{result.stderr}")
            # except Exception as e:
            #     print(e)

if __name__ == '__main__':
    run_training()
