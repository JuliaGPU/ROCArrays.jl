#!/bin/bash

set -eu

# Shell script to transform CuArrays.jl into ROCArrays.jl

CUARRAYS_PATH=$1
LASTWD=$PWD

mkdir -p ./temp
cd ./temp

if [ -d ./CuArrays ]; then
    rm -rf ./CuArrays
fi
cp -aR $CUARRAYS_PATH ./CuArrays
if [ -d ./ROCArrays ]; then
    rm -rf ./ROCArrays
fi
mkdir ./ROCArrays
cd ./CuArrays

# Fill in directory structure
while read -r dir; do
    echo "Creating $dir"
    mkdir ../ROCArrays/$dir
done < <(find src/ test/ -type d)

# Mass search-replace
while read -r file; do
    CUPATH=$file
    ROCPATH=../ROCArrays/$file
    echo "Processing $CUPATH => $ROCPATH"
    cp $CUPATH $ROCPATH

    sed -i 's/CuArray/ROCArray/g' $ROCPATH
    sed -i 's/CUDAnative/AMDGPUnative/g' $ROCPATH
    sed -i 's/CUDAdrv/HSARuntime/g' $ROCPATH
    sed -i 's/CUDAapi//g' $ROCPATH

    sed -i 's/CUBLAS/rocBLAS/g' $ROCPATH
    sed -i 's/CUSPARSE/rocSPARSE/g' $ROCPATH
    sed -i 's/CUSOLVER/rocALUTION/g' $ROCPATH
    sed -i 's/CUFFT/rocFFT/g' $ROCPATH
    sed -i 's/CURAND/rocRAND/g' $ROCPATH
    sed -i 's/CUDNN/MIOpen/g' $ROCPATH
    sed -i 's/CUTENSOR/MIOpen/g' $ROCPATH

    sed -i 's/cuones/rocones/g' $ROCPATH
    sed -i 's/cuzeros/roczeros/g' $ROCPATH
    sed -i 's/cufill/rocfill/g' $ROCPATH

    sed -i 's/CUDA/ROC/g' $ROCPATH
    sed -i 's/cuda/roc/g' $ROCPATH
    sed -i 's/CU/ROC/g' $ROCPATH
    sed -i 's/Cu/ROC/g' $ROCPATH
    sed -i 's/cu/roc/g' $ROCPATH
done < <(find src/ test/ -type f -iname '*.jl')

# Leave CuArrays and enter ROCArrays
cd ../ROCArrays

# Manual tweaks
mv src/CuArrays.jl src/ROCArrays.jl

# Apply patches
for patchfile in $LASTWD/patches; do
    patch < $patchfile
done

# Make ROCArrays folder testable
cp $LASTWD/Project.toml ./ROCArrays/
cp $LASTWD/Manifest.toml ./ROCArrays/

cd $LASTWD
