#!/bin/bash
#BSUB -J early_fusion_training                # Job name
#BSUB -q gpuv100                     # Queue to submit to
#BSUB -W 20                     # Wall time limit (20 minutes)
#BSUB -R "rusage[mem=2GB]"       # Memory requirement (5GB)
#BSUB -n 4                        # Number of CPUs (cores) - usually 1 is enough for GPU jobs
#BSUB -R "span[hosts=1]"           # All CPUs on a single host
#BSUB -gpu "num=1:mode=exclusive_process" # Request 1 GPU in exclusive_process mode
#BSUB -o early_fusion_training_%J.out
#BSUB -e early_fusion_training_%J.err

# change to right directory

cd ~/Documents/Python/Introduction_to_Deep_Learning_for_CV/Assignments_DL_for_CV/

# Activate your virtual environment
source /zhome/b4/e/214014/Documents/Python/Introduction_to_Deep_Learning_for_CV/venv_DL_CV/bin/activate

# Run your Python script
python /zhome/b4/e/214014/Documents/Python/Introduction_to_Deep_Learning_for_CV/Assignments_DL_for_CV/Assignment_2/early_fusion.py