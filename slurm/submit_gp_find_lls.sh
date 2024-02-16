#!/bin/bash

# Example usage of this script:
# NSPEC=10
# sbatch --export=MAX_LYA=5,NUM_LINES=4,IMG_DIR="images-lls/",LLS_SAMPLE_H5="data/dr12q/processed/lls_samples.h5",NSPEC=$NSPEC --output="gp_find_lls_nspec-${NSPEC}-%J.out" your_slurm_script.sh

# Example loop to submit multiple jobs with different NSPEC values:
# # Loop from 0 to 1000
# for NSPEC in {0..1000}; do
#   sbatch --export=ALL,NSPEC=$NSPEC,MAX_LYA=5,NUM_LINES=4,IMG_DIR="images-lls/",LLS_SAMPLE_H5="data/dr12q/processed/lls_samples.h5" \
#          --output="gp_find_lls_nspec-${NSPEC}-%J.out" \
#          your_slurm_script.sh
# done

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --job-name=gp_find_lls_nspec
#SBATCH -p short

echo "Running gp_find_lls.py with the following parameters:"
echo "NSPEC = $NSPEC"
echo "MAX_LYA = $MAX_LYA"
echo "NUM_LINES = $NUM_LINES"
echo "IMG_DIR = $IMG_DIR"
echo "LLS_SAMPLE_H5 = $LLS_SAMPLE_H5"
echo "----"

# run python script with variable arguments
/rhome/mho026/.conda/envs/fast-mpi4py/bin/python -u examples/gp_find_lls.py \
    --nspec $NSPEC \
    --max_Lya $MAX_LYA \
    --num_lines $NUM_LINES \
    --img_dir $IMG_DIR \
    --lls_sample_h5 $LLS_SAMPLE_H5

hostname



exit

