#!/bin/bash
#BSUB -J single_frame_CNN_training            # Job name
#BSUB -q gpuv100                              # Queue to submit to (V100 GPUs)
#BSUB -W 0:20                                 # Wall time limit (20 minutes)
#BSUB -R "rusage[mem=5GB]"                    # Memory requirement (5GB)
#BSUB -n 4                                    # Number of CPU cores
#BSUB -R "span[hosts=1]"                      # All CPUs on a single host
#BSUB -gpu "num=1:mode=exclusive_process"     # Request 1 GPU in exclusive mode
#BSUB -o single_frame_CNN_training_%J.out     # Standard output file
#BSUB -e single_frame_CNN_training_%J.err     # Error output file

# --- Load CUDA environment (required for GPU access) ---
module load cuda/11.8

# --- Print environment info for debugging/logging ---
echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# --- Move to working directory ---
cd ~/Documents/Python/Introduction_to_Deep_Learning_for_CV/Assignments_DL_for_CV/ || exit 1

# --- Activate virtual environment ---
source ~/Documents/Python/Introduction_to_Deep_Learning_for_CV/venv_DL_CV/bin/activate

# --- Run your Python training script ---
python /zhome/b4/e/214014/Documents/Python/Introduction_to_Deep_Learning_for_CV/Assignments_DL_for_CV/Assignment_2/task4.2.1.py
