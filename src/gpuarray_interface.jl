import GPUArrays

struct ROCArrayBackend <: GPUArrays.GPUBackend end
GPUArrays.backend(::Type{<:ROCArray}) = ROCArrayBackend()


#Abstract GPU interface
struct ROCKernelState end

@inline function GPUArrays.LocalMemory(::ROCKernelState, ::Type{T}, ::Val{N}, ::Val{id}) where {T, N, id}
    ptr = AMDGPUnative._shmem(Val(id), T, Val(N))
    ROCDeviceArray(N, DevicePtr{T, AMDGPUnative.AS.Shared}(ptr))
end

GPUArrays.AbstractDeviceArray(A::AMDGPUnative.ROCDeviceArray, shape) = AMDGPUnative.ROCDeviceArray(shape, pointer(A))

@inline GPUArrays.synchronize_threads(::ROCKernelState) = AMDGPUnative.sync_threads()

GPUArrays.blas_module(::ROCArray) = ROCArrays.rocBLAS
GPUArrays.blasbuffer(x::ROCArray) = x

"""
Blocks until all operations are finished on `A`
"""
GPUArrays.synchronize(A::ROCArray) =
    CUDAdrv.synchronize()

for (i, sym) in enumerate((:x, :y, :z))
    for (f, fcu) in (
            (:blockidx, :blockIdx),
            (:blockdim, :blockDim),
            (:threadidx, :threadIdx),
            (:griddim, :gridDim)
        )
        fname = Symbol(string(f, '_', sym))
        rocfun = Symbol(string(fcu, '_', sym))
        @eval GPUArrays.$fname(::ROCKernelState) = AMDGPUnative.$rocfun()
    end
end

# devices() = CUDAdrv.devices()
GPUArrays.device(A::ROCArray) = CUDAdrv.device(CUDAdrv.ROCCurrentContext())
GPUArrays.is_gpu(agent::HSAAgent) = true
GPUArrays.name(agent::HSAAgent) = string("CU ", CUDAdrv.name(agent))
GPUArrays.threads(agent::HSAAgent) = CUDAdrv.attribute(agent, CUDAdrv.MAX_THREADS_PER_BLOCK)

GPUArrays.blocks(agent::HSAAgent) =
    (CUDAdrv.attribute(agent, CUDAdrv.MAX_BLOCK_DIM_X),
     CUDAdrv.attribute(agent, CUDAdrv.MAX_BLOCK_DIM_Y),
     CUDAdrv.attribute(agent, CUDAdrv.MAX_BLOCK_DIM_Z))

GPUArrays.free_global_memory(agent::HSAAgent) = CUDAdrv.Mem.info()[1]
GPUArrays.global_memory(agent::HSAAgent) = CUDAdrv.totalmem(agent)
GPUArrays.local_memory(agent::HSAAgent) = CUDAdrv.attribute(agent, CUDAdrv.TOTAL_CONSTANT_MEMORY)

function GPUArrays._gpu_call(::ROCArrayBackend, f, A, args::Tuple,
                             blocks_threads::Tuple{T, T}) where {N, T <: NTuple{N, Integer}}
    blk, thr = blocks_threads
    @roc gridsize=blk groupsize=thr f(ROCKernelState(), args...)
end

# Save reinterpret and reshape implementation use this in GPUArrays
GPUArrays.unsafe_reinterpret(::Type{T}, A::ROCArray, size::NTuple{N, Integer}) where {T, N} =
    ROCArray{T, N}(A.buf, size)
