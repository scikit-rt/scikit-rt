#!/bin/bash
HOME=/usera/harrison
cd ${HOME}/codeshare/nnunet
source ${HOME}/bin/conda-setup.sh

conda activate nnunet
export nnUNet_raw=$(pwd)/nnUNet_raw
export nnUNet_preprocessed=$(pwd)/nnUNet_preprocessed
export nnUNet_results=$(pwd)/nnUNet_results

mkdir -p ${nnUNet_raw}
mkdir -p ${nnUNet_preprocessed}
mkdir -p ${nnUNet_results}

nnUNetv2_train 1 3d_fullres all -device cpu
