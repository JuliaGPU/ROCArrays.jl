module rocFFT

using ..ROCArrays
using ..ROCArrays: librocfft
using ..HIP: hipStreamSynchronize

using CEnum

include("librocfft_common.jl")
include("error.jl")
version() = VersionNumber(rocfft_version_major, rocfft_version_minor, rocfft_version_patch)

include("librocfft.jl")
include("util.jl")
include("fft.jl")

rocfft_setup()
atexit(rocfft_cleanup)


end
