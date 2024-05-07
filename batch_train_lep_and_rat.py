# Define the different configurations for label error percent and ratio

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
    #batch_size = 32
    batch_size = 64
 
    # Define the different configurations for learning rates and batch sizes
    leps = np.linspace(0.0, 1.0, 11)
    ratios = np.linspace(-1.5, 1.5, 13)

    # Define the constant parameters
    dataset = "seg_isic"
    overwrite = "--overwrite"
    #workers = "--workers 0"

    # Iterate over each combination of label error percent and ratio
    for lep in leps:
        for ratio in ratios:
            log_name = f"isic_unet_{learning_rate}_{batch_size}_{lep}_{ratio}"
            command = [
                #sys.executable, "train.py",
                venv_python, "train.py",
                "--dataset", dataset,
                "--log_name", log_name,
                overwrite,
                "--lr", learning_rate,
                "--batch-size", str(batch_size),
                "--label_error_percent", str(lep),
                "--ratio", str(ratio),
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
