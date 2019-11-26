#!/bin/bash

set -eu

# Shell script to transform CuArrays.jl into ROCArrays.jl

CUARRAYS_PATH=$1

if [ -d ./CuArrays ]; then
    rm -rf ./CuArrays
fi
cp -aR $CUARRAYS_PATH ./CuArrays
cd ./CuArrays
mkdir -p ./temp

while read -r file; do
    CUPATH=$file
    TMPPATH=temp/$file
    echo $CUPATH "=>" $TMPPATH
    continue
    sed 's/CuArray/ROCArray/g' CuArrays/$file > ./ROCArrays/$file
    sed 's/CUDAnative/AMDGPUnative/g' CuArrays/$file > ./ROCArrays/$file
    sed 's/CUDAdrv/HSARuntime/g' CuArrays/$file > ./ROCArrays/$file
    sed 's/CUDAapi//g' CuArrays/$file > ./ROCArrays/$file

    sed 's/CUBLAS/rocBLAS/g' CuArrays/$file > ./ROCArrays/$file
    sed 's/CUSPARSE/rocSPARSE/g' CuArrays/$file > ./ROCArrays/$file
    sed 's/CUSOLVER/rocALUTION/g' CuArrays/$file > ./ROCArrays/$file
    sed 's/CUFFT/rocFFT/g' CuArrays/$file > ./ROCArrays/$file
    sed 's/CURAND/rocRAND/g' CuArrays/$file > ./ROCArrays/$file
    sed 's/CUDNN/MIOpen/g' CuArrays/$file > ./ROCArrays/$file
    sed 's/CUTENSOR/MIOpen/g' CuArrays/$file > ./ROCArrays/$file

    sed 's/cuones/rocones/g' CuArrays/$file > ./ROCArrays/$file
    sed 's/cuzeros/roczeros/g' CuArrays/$file > ./ROCArrays/$file
    sed 's/cufill/rocfill/g' CuArrays/$file > ./ROCArrays/$file

    sed 's/CUDA/ROC/g' CuArrays/$file > ./ROCArrays/$file
    sed 's/cuda/roc/g' CuArrays/$file > ./ROCArrays/$file
    sed 's/CU/ROC/g' CuArrays/$file > ./ROCArrays/$file
    sed 's/Cu/ROC/g' CuArrays/$file > ./ROCArrays/$file
    sed 's/cu/roc/g' CuArrays/$file > ./ROCArrays/$file
done < <(find src/ test/ -type f -iname '*.jl')

cd ..
