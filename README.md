# ROCArrays - Julia Arrays for Radeon/AMD GPUs

### Warning

ROCArrays is currently an experimental package, and many things are broken or
not implemented.

### Overview

ROCArrays is a cousin package to CuArrays, but instead provides array
operations for AMD GPUs running on the ROCm stack. We strive to match
CuArray's API to make it easy for users to use either stack with minimal code
changes.

### Requirements

ROCArrays optionally depends on ROCm "external" libraries for various
features, e.g. BLAS support depends on rocBLAS, neural network support depends
on MIOpen, etc. If such libraries are installed appropriately on your system,
then ROCArrays will enable the appropriate Julia methods. On Ubuntu, the
following packages should be installed for full support:
`rocm-dev rocblas rocsparse rocfft rocrand rocalution miopen-opencl`

### Acknowledgements

ROCArrays builds on the excellent work done by Tim Besard and contributors to
CuArrays.jl, and would not exist in a working state without their hard work.
