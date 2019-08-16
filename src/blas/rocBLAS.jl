module rocBLAS

#import CUDAdrv: CUDAdrv, CuContext, CuStream_t
#import CUDAapi

using ..ROCArrays
#using ..ROCArrays: libcublas, active_context

using LinearAlgebra

include("librocblas_types.jl")
include("error.jl")

#=
const _handles = Dict{CuContext,cublasHandle_t}()
const _handle = Ref{cublasHandle_t}(C_NULL)

function handle()
    if _handle[] == C_NULL
        @assert isassigned(active_context) # some other call should have initialized CUDA
        _handle[] = get!(_handles, active_context[]) do
            context = active_context[]
            handle = cublasCreate_v2()

            # enable tensor math mode if our device supports it, and fast math is enabled
            dev = CUDAdrv.device(context)
            if Base.JLOptions().fast_math == 1 && CUDAdrv.capability(dev) >= v"7.0"
              cublasSetMathMode(CUBLAS_TENSOR_OP_MATH, handle)
            end

            atexit(()->CUDAdrv.isvalid(context) && cublasDestroy_v2(handle))
            handle
        end
    end

    return _handle[]
end
=#

include("librocblas.jl")
#=
include("util.jl")
include("wrappers.jl")
include("highlevel.jl")

version() = VersionNumber(cublasGetProperty(CUDAapi.MAJOR_VERSION),
                          cublasGetProperty(CUDAapi.MINOR_VERSION),
                          cublasGetProperty(CUDAapi.PATCH_LEVEL))
=#
end
