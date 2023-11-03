#!/usr/bin/env fish

# Set the path to SPlishSPlasHs DynamicBoundarySimulator in splishsplash_config.py
# before running this script

# output directories
set -lx OUTPUT_SCENES_DIR "/Users/bekzatajan/PhD/Thesis/mldem_rebuilt/data/exp_1"
set -lx OUTPUT_DATA_DIR "$OUTPUT_SCENES_DIR""_data"

if test -e $OUTPUT_DATA_DIR
    rm -r $OUTPUT_DATA_DIR
end

# Transforms and compresses the data such that it can be used for training.
# This will also create the OUTPUT_DATA_DIR.
python3 create_physics.py --input $OUTPUT_SCENES_DIR  --output $OUTPUT_DATA_DIR  
#python3 create_physics_records_mfix_mew.py --input $OUTPUT_SCENES_DIR  --output $OUTPUT_DATA_DIR  

