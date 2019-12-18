# Modified from GPUArrays.jl/src/array.jl (JLArray)

using AbstractFFTs
import Base.Broadcast: BroadcastStyle, Broadcasted, ArrayStyle
import GPUArrays
import GPUArrays: GPUArray, GPUBackend, LocalMemory, @allowscalar, allowscalar

using LinearAlgebra

import HSARuntime: hsa_memory_allocate, hsa_memory_free, check, DEFAULT_AGENT
import AMDGPUnative: DevicePtr

mutable struct ROCArray{T,N} <: GPUArray{T,N}
    buf::Mem.Buffer
    own::Bool

    dims::Dims{N}
    offset::Int

    function ROCArray{T,N}(buf::Mem.Buffer, dims::Dims{N}; offset::Integer=0, own::Bool=true) where {T,N}
        xs = new{T,N}(buf, own, dims, offset)
        if own
            Mem.retain(buf)
            finalizer(unsafe_free!, xs)
        end
        return xs
    end
end

function unsafe_free!(xs::ROCArray)
    Mem.release(xs.buf) && Mem.free(xs.buf)
    return
end

const ROCVector{T} = ROCArray{T,1}
const ROCMatrix{T} = ROCArray{T,2}
const ROCVecOrMat{T} = Union{ROCVector{T},ROCMatrix{T}}

## construction

# type and dimensionality specified, accepting dims as tuples of Ints
ROCArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N} =
  ROCArray{T,N}(Mem.alloc(prod(dims)*sizeof(T)), dims)

# type and dimensionality specified, accepting dims as series of Ints
ROCArray{T,N}(::UndefInitializer, dims::Integer...) where {T,N} = ROCArray{T,N}(undef, dims)

# type but not dimensionality specified
ROCArray{T}(::UndefInitializer, dims::Dims{N}) where {T,N} = ROCArray{T,N}(undef, dims)
ROCArray{T}(::UndefInitializer, dims::Integer...) where {T} =
  ROCArray{T}(undef, convert(Tuple{Vararg{Int}}, dims))

# empty vector constructor
ROCArray{T,1}() where {T} = ROCArray{T,1}(undef, 0)


Base.similar(a::ROCArray{T,N}) where {T,N} = ROCArray{T,N}(undef, size(a))
Base.similar(a::ROCArray{T}, dims::Base.Dims{N}) where {T,N} = ROCArray{T,N}(undef, dims)
Base.similar(a::ROCArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} = ROCArray{T,N}(undef, dims)


## array interface

Base.elsize(::Type{<:ROCArray{T}}) where {T} = sizeof(T)

Base.size(x::ROCArray) = x.dims
Base.sizeof(x::ROCArray) = Base.elsize(x) * length(x)


## interop with other arrays

ROCArray{T,N}(x::AbstractArray{S,N}) where {T,N,S} =
    ROCArray{T,N}(convert(Array{T}, x), size(x))
function ROCArray{T,N}(x::Array{T,N}) where {T,N}
    r = ROCArray{T,N}(undef, size(x))
    Mem.upload!(r.buf, pointer(x), sizeof(x))
    return r
end

# underspecified constructors
ROCArray{T}(xs::AbstractArray{S,N}) where {T,N,S} = ROCArray{T,N}(xs)
(::Type{ROCArray{T,N} where T})(x::AbstractArray{S,N}) where {S,N} = ROCArray{S,N}(x)
ROCArray(A::AbstractArray{T,N}) where {T,N} = ROCArray{T,N}(A)

# idempotency
ROCArray{T,N}(xs::ROCArray{T,N}) where {T,N} = xs


## conversions

Base.pointer(arr::ROCArray{T,N}) where {T,N} =
    Base.unsafe_convert(Ptr{T}, arr.buf.ptr)
Base.convert(::Type{T}, x::T) where T<:ROCArray = x
#Base.convert(::Type{<:AbstractArray{T,N}}, x::ROCArray{T,N}) where {T,N} = A(x)
Base.collect(x::ROCArray{T,N}) where {T,N} = copyto!(Array{T,N}(undef, size(x)), x)
Base.unsafe_convert(::Type{Ptr{T}}, x::ROCArray{T}) where T =
    Base.unsafe_convert(Ptr{T}, x.buf)

## broadcast

BroadcastStyle(::Type{<:ROCArray}) = ArrayStyle{ROCArray}()

function Base.similar(bc::Broadcasted{ArrayStyle{ROCArray}}, ::Type{T}) where T
    similar(ROCArray{T}, axes(bc))
end

Base.similar(bc::Broadcasted{ArrayStyle{ROCArray}}, ::Type{T}, dims...) where {T} = ROCArray{T}(undef, dims...)

## gpuarray interface

struct ROCBackend <: GPUBackend end
GPUArrays.backend(::Type{<:ROCArray}) = ROCBackend()


#=
"""
Thread group local memory
"""
struct LocalMem{N, T}
    x::NTuple{N, Vector{T}}
end

to_device(state, x::ROCArray) = x.data
to_device(state, x::Tuple) = to_device.(Ref(state), x)
to_device(state, x::Base.RefValue{<: ROCArray}) = Base.RefValue(to_device(state, x[]))
to_device(state, x) = x
# creates a `local` vector for each thread group
to_device(state, x::LocalMemory{T}) where T = LocalMem(ntuple(i-> Vector{T}(x.size), GPUArrays.blockdim_x(state)))

to_blocks(state, x) = x
# unpacks local memory for each block
to_blocks(state, x::LocalMem) = x.x[GPUArrays.blockidx_x(state)]

GPUArrays.unsafe_reinterpret(::Type{T}, A::ROCArray, size::Tuple) where T =
    reshape(reinterpret(T, A.data), size)
=#

function Base.copyto!(dest::Array{T}, d_offset::Integer,
                      source::ROCArray{T}, s_offset::Integer,
                      amount::Integer) where T
    @boundscheck checkbounds(dest, d_offset+amount-1)
    @boundscheck checkbounds(source, s_offset+amount-1)
    #copyto!(dest, d_offset, source.data, s_offset, amount)
    Mem.download!(pointer(dest, d_offset),
                  Mem.view(source.buf, (s_offset-1)*sizeof(T)),
                  amount*sizeof(T))
    dest
end
function Base.copyto!(dest::ROCArray{T}, d_offset::Integer,
                      source::Array{T}, s_offset::Integer,
                      amount::Integer) where T
    @boundscheck checkbounds(dest, d_offset+amount-1)
    @boundscheck checkbounds(source, s_offset+amount-1)
    #copyto!(dest.data, d_offset, source, s_offset, amount)
    Mem.upload!(Mem.view(dest.buf, (d_offset-1)*sizeof(T)),
                pointer(source, s_offset),
                amount*sizeof(T))
    dest
end
function Base.copyto!(dest::ROCArray{T}, d_offset::Integer,
                      source::ROCArray{T}, s_offset::Integer,
                      amount::Integer) where T
    @boundscheck checkbounds(dest, d_offset+amount-1)
    @boundscheck checkbounds(source, s_offset+amount-1)
    #copyto!(dest.data, d_offset, source.data, s_offset, amount)
    Mem.transfer!(Mem.view(dest.buf, (d_offset-1)*sizeof(T)),
                  Mem.view(source.buf, (s_offset-1)*sizeof(T)),
                  amount*sizeof(T))
    dest
end

struct ROCKernelState end

@inline function GPUArrays.LocalMemory(::ROCKernelState, ::Type{T}, ::Val{N}, ::Val{id}
                                      ) where {T, N, id}
    ptr = AMDGPUnative._shmem(Val(id), T, Val(prod(N)))
    ROCDeviceArray(N, DevicePtr{T, AMDGPUnative.AS.Shared}(ptr))
end

GPUArrays.AbstractDeviceArray(A::AMDGPUnative.ROCDeviceArray, shape) =
    AMDGPUnative.ROCDeviceArray(shape, pointer(A))

@inline function GPUArrays.synchronize_threads(::ROCKernelState)
    AMDGPUnative.sync_workgroup()
    return
end

#= FIXME: Use HIP to sync this device's queues
"""
Blocks until all operations are finished on `A`
"""
GPUArrays.synchronize(A::ROCArray) =
    sync_queues()
=#

for (i, sym) in enumerate((:x, :y, :z))
    for (f, froc) in (
            (:blockidx, :blockIdx),
            (:blockdim, :blockDim),
            (:threadidx, :threadIdx),
            (:griddim, :gridDim)
        )
        fname = Symbol(string(f, '_', sym))
        rocfun = Symbol(string(froc, '_', sym))
        @eval GPUArrays.$fname(::ROCKernelState) = AMDGPUnative.$rocfun()
    end
end

GPUArrays.blas_module(::ROCArray) = rocBLAS
GPUArrays.blasbuffer(A::ROCArray) = A

GPUArrays.device(A::ROCArray) = A.buf.agent
GPUArrays.is_gpu(agent::HSAAgent) = true # FIXME: Not always true
GPUArrays.name(agent::HSAAgent) = HSARuntime.get_name(agent)

#= FIXME
GPUArrays.threads(agent::HSAAgent) =
    CUDAdrv.attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)

GPUArrays.blocks(dev::CUDAdrv.CuDevice) =
    (CUDAdrv.attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X),
     CUDAdrv.attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y),
     CUDAdrv.attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z))

GPUArrays.free_global_memory(dev::CUDAdrv.CuDevice) = CUDAdrv.Mem.info()[1]
GPUArrays.global_memory(dev::CUDAdrv.CuDevice) = CUDAdrv.totalmem(dev)
GPUArrays.local_memory(dev::CUDAdrv.CuDevice) =
    CUDAdrv.attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY)
=#

function GPUArrays._gpu_call(::ROCBackend, f, A, args::Tuple,
                             blocks_threads::Tuple{T, T}) where {N, T <: NTuple{N, Integer}}
    blk, thr = blocks_threads
    error("Not implemented")
    @roc groupsize=blk gridsize=thr f(ROCKernelState(), args...)
end

# Save reinterpret and reshape implementation use this in GPUArrays
function GPUArrays.unsafe_reinterpret(::Type{T}, A::ROCArray,
                                      size::NTuple{N, Integer}) where {T, N}

    return ROCArray{T,N}(convert(Ptr{T}, A), size, A)
end

#= FIXME
# defining our own plan type is the easiest way to pass around the plans in FFTW interface
# without ambiguities
struct FFTPlan{T}
    p::T
end

AbstractFFTs.plan_fft(A::ROCArray; kw_args...) = FFTPlan(plan_fft(A.data; kw_args...))
AbstractFFTs.plan_fft!(A::ROCArray; kw_args...) = FFTPlan(plan_fft!(A.data; kw_args...))
AbstractFFTs.plan_bfft!(A::ROCArray; kw_args...) = FFTPlan(plan_bfft!(A.data; kw_args...))
AbstractFFTs.plan_bfft(A::ROCArray; kw_args...) = FFTPlan(plan_bfft(A.data; kw_args...))
AbstractFFTs.plan_ifft!(A::ROCArray; kw_args...) = FFTPlan(plan_ifft!(A.data; kw_args...))
AbstractFFTs.plan_ifft(A::ROCArray; kw_args...) = FFTPlan(plan_ifft(A.data; kw_args...))

function Base.:(*)(plan::FFTPlan, A::ROCArray)
    x = plan.p * A.data
    ROCArray(x)
end
=#
