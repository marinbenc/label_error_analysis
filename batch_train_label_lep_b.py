# Define the different configurations for label error percent and bias

import subprocess
import sys
import os
import numpy as np

def run_training():

    venv_python = "C:\\Git\\label_error_analysis\\env\\Scripts\\python.exe"
    #print("Using Python executable:", sys.executable)
    print("Using Python executable:", venv_python)

    # Define the different configurations for learning rates and batch sizes
    learning_rate = "1e-4"
    batch_sizes = [64, 32]
     
    # Define the different configurations for label error percent and bias
    leps = [0.0, 0.25, 0.50, 0.75, 1.0]
    biases = [-1, 0, 1]

    # Define the constant parameters
    dataset = "seg_isic"
    overwrite = "--overwrite"
    #workers = "--workers 0"

    # Iterate over each combination of batch size, label error percent and ratio
    for batch_size in batch_sizes:
        for lep in leps:
            for bias in biases:
                log_name = f"isic_unet_{learning_rate}_{batch_size}_{int(lep * 100)}_{bias}"
                command = [
                    #sys.executable, "train.py",
                    venv_python, "train.py",
                    "--dataset", dataset,
                    "--log_name", log_name,
                    overwrite,
                    "--lr", learning_rate,
                    "--batch-size", str(batch_size),
                    "--label_error_percent", str(lep),
                    "--bias", str(bias),
                    "--folds", str(5),
                    #workers
                ]

                print(f"Running: {' '.join(command)}")
                env = os.environ.copy()
                result = subprocess.run(command, text=True, capture_output=True, encoding='utf-8', shell=False, env=env)

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
