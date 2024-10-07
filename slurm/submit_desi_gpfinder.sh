#!/bin/bash
#SBATCH --job-name=${JOB_NAME:-dla_detection}     # Job name, default is "dla_detection"
#SBATCH --output=${OUTPUT_FILE:-gpdla_%j.log}     # Standard output log, default is "gpdla_%j.log"
#SBATCH --error=${ERROR_FILE:-error_%j.log}       # Standard error log, default is "error_%j.log"
#SBATCH --ntasks=${NTASKS:-32}                   # Number of tasks, default is 32
#SBATCH --mem=${MEMORY:-64G}                      # Memory allocation, default is 64 GB
#SBATCH --time=${TIME:-24:00:00}                  # Time limit, default is 24 hours
#SBATCH --partition=${PARTITION:-normal}          # Partition name, default is "normal"

# Comments:
# The --ntasks parameter specifies the number of independent processes that will be run.
# Modify --ntasks based on your job requirements.

# Load DESI specific modules/environment
source /global/cfs/cdirs/desi/software/desi_environment.sh 23.1

# Execute the Python script with all arguments
python run_bayes_select.py \
    --spectra_filename ${SPECTRA_FILENAME:-/path/to/spectra.fits} \          # Path to the spectra file
    --zbest_filename ${ZBEST_FILENAME:-/path/to/zbest.fits} \                # Path to the redshift catalog
    --learned_file ${LEARNED_FILE:-data/dr12q/processed/learned_qso_model_lyseries_variance.mat} \  # QSO model file path
    --catalog_name ${CATALOG_NAME:-data/dr12q/processed/catalog.mat} \       # Catalog file path
    --los_catalog ${LOS_CATALOG:-data/dla_catalogs/processed/los_catalog} \  # Line-of-sight catalog path
    --dla_catalog ${DLA_CATALOG:-data/dla_catalogs/processed/dla_catalog} \  # DLA catalog path
    --dla_samples_file ${DLA_SAMPLES_FILE:-data/dr12q/processed/dla_samples_a03.mat} \              # DLA samples file
    --sub_dla_samples_file ${SUB_DLA_SAMPLES_FILE:-data/dr12q/processed/subdla_samples.mat} \        # Sub-DLA samples file
    --min_z_separation ${MIN_Z_SEPARATION:-3000.0} \                         # Minimum redshift separation for DLAs
    --prev_tau_0 ${PREV_TAU_0:-0.00554} \                                    # Previous tau_0 value
    --prev_beta ${PREV_BETA:-3.182} \                                        # Previous beta value
    --max_dlas ${MAX_DLAS:-3} \                                              # Maximum number of DLAs to model
    --plot_figures ${PLOT_FIGURES:-0} \                                      # Set to 1 to generate plots
    --max_workers ${MAX_WORKERS:-32} \                                       # Number of workers for parallel processing
    --batch_size ${BATCH_SIZE:-313} \                                        # Batch size for parallel computation
    --loading_min_lambda ${LOADING_MIN_LAMBDA:-800} \                        # Rest wavelengths to load (Å)
    --loading_max_lambda ${LOADING_MAX_LAMBDA:-1550} \                       # Rest wavelengths to load (Å)
    --normalization_min_lambda ${NORMALIZATION_MIN_LAMBDA:-1425} \           # Rest wavelengths for normalization
    --normalization_max_lambda ${NORMALIZATION_MAX_LAMBDA:-1475} \           # Rest wavelengths for normalization
    --min_lambda ${MIN_LAMBDA:-850.75} \                                     # Rest wavelengths to model (Å)
    --max_lambda ${MAX_LAMBDA:-1420.75} \                                    # Rest wavelengths to model (Å)
    --dlambda ${DLAMBDA:-0.25} \                                             # Separation of wavelength grid (Å)
    --k ${K:-20} \                                                           # Rank of non-diagonal contribution
    --max_noise_variance ${MAX_NOISE_VARIANCE:-9}                            # Maximum pixel noise for training

# Final notes:
# - You can customize any of the script arguments using the `sbatch` command.
# - Example: sbatch --export=ALL,NTASKS=16,SPECTRA_FILENAME=/path/to/spectra.fits submit_script.sh
# - Ensure the number of workers (max_workers) is aligned with the number of tasks (ntasks).

# exit the script
exit 0
