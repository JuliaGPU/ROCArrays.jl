module HSAArrays

using AMDGPUnative
import HSARuntime: HSAArray, HSAAgent

using GPUArrays

export HSAArray, HSAVector, HSAMatrix, HSAVecOrMat
export hsa, hsazeros, hsaones, hsafill

import LinearAlgebra

using Adapt

const ext = joinpath(dirname(@__DIR__), "deps", "ext.jl")
isfile(ext) || error("HSAArrays.jl has not been built, please run Pkg.build(\"HSAArrays\").")
include(ext)
if !configured
    # default (non-functional) values for critical variables,
    # making it possible to _load_ the package at all times.
    const librocblas = nothing
    const librocsparse = nothing
    const librocalution = nothing
    const librocfft = nothing
    const librocrand = nothing
    const libmiopen = nothing
end

#=
include("memory.jl")
include("array.jl")
include("subarray.jl")
include("utils.jl")
include("indexing.jl")
include("broadcast.jl")
include("matmul.jl")
include("mapreduce.jl")
=#

#include("gpuarray_interface.jl")

# many libraries need to be initialized per-device (per-context, really, but
# we assume users of HSAArrays and/or AMDGPUnative only use a single context),
# so keep track of the active one.
#const active_context = Ref{ROCContext}()

librocblas !== nothing    && include("blas/rocBLAS.jl")
librocsparse !== nothing  && include("sparse/rocSPARSE.jl")
librocalution !== nothing && include("solver/rocALUTION.jl")
librocfft !== nothing     && include("fft/rocFFT.jl")
librocrand !== nothing    && include("rand/rocRAND.jl")
libmiopen !== nothing     && include("dnn/MIOpen.jl")

#include("nnlib.jl")

#include("deprecated.jl")

function __init__()
    if !configured
        @warn("HSAArrays.jl has not been successfully built, and will not work properly.")
        @warn("Please run Pkg.build(\"HSAArrays\") and restart Julia.")
        return
    end

    function check_library(name, path)
        path === nothing && return
        if !ispath(path)
            error("$name library has changed. Please run Pkg.build(\"HSAArrays\") and restart Julia.")
        end
    end
    check_library("rocBLAS", librocblas)
    check_library("rocSPARSE", librocsparse)
    check_library("rocALUTION", librocalution)
    check_library("rocFFT", librocfft)
    check_library("rocRAND", librocrand)
    check_library("MIOpen", libmiopen)

    # update the active context when we switch devices
    #= FIXME
    callback = (::HSAAgent, ctx::ROCContext) -> begin
        active_context[] = ctx

        # wipe the active handles
        isdefined(HSAArrays, :rocBLAS)   && (rocBLAS._handle[] = C_NULL)
        isdefined(HSAArrays, :rocSOLVER) && (rocSOLVER._dense_handle[] = C_NULL)
        isdefined(HSAArrays, :rocRAND)   && (rocRAND._generator[] = nothing)
        isdefined(HSAArrays, :MIOpen)    && (MIOpen._handle[] = C_NULL)
    end
    push!(AMDGPUnative.device!_listeners, callback)

    # a device might be active already
    existing_ctx = CUDAdrv.CuCurrentContext()
    if existing_ctx !== nothing
        active_context[] = existing_ctx
    end

    __init_memory__()
    =#
end

end # module
