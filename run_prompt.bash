#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -J meta-learning-synthetic-data
#SBATCH --output=/home/hgf_hmgu/hgf_tfv0045/meta-learning-synthetic-data/outputs/%x.out
#SBATCH --error=/home/hgf_hmgu/hgf_tfv0045/meta-learning-synthetic-data/outputs/%x.err
#SBATCH --partition=normal
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:4


# Your other commands
CODE_PATH="/home/hgf_hmgu/hgf_tfv0045/meta-learning-synthetic-data"
OUT_PATH="/home/hgf_hmgu/hgf_tfv0045/meta-learning-synthetic-data/outputs"

# Check if the directory exists
if [ ! -d "$OUT_PATH" ]; then
    # Directory does not exist, so create it
    mkdir -p "$OUT_PATH"
    echo "Directory created: $OUT_PATH"
else
    echo "Directory already exists: $OUT_PATH"
fi

source ~/.bash_profile


# Activate your environment
source env/bin/activate


# Run your Python script
srun python ${CODE_PATH}/src/synth_data_prompt.py 
