#!/bin/sh
### Queue
#BSUB -q c02516
### Request 1 GPU
#BSUB -gpu "num=1:mode=exclusive_process"
### Job name
#BSUB -J testjob
### Number of cores
#BSUB -n 4
#BSUB -R "span[hosts=1]"
### CPU memory
#BSUB -R "rusage[mem=20GB]"
### Wall-clock time (max 12 hours)
#BSUB -W 12:00
### Output and error files
#BSUB -o OUTPUT_FILE_%J.out
#BSUB -e OUTPUT_FILE_%J.err

# Activate your virtual environment
source /zhome/b4/e/214014/Documents/Python/Introduction_to_Deep_Learning_for_CV/venv_DL_CV/bin/activate

# Run your Python script
python /zhome/b4/e/214014/Documents/Python/Introduction_to_Deep_Learning_for_CV/Assignments_DL_for_CV/Assignment_2/datasets.py