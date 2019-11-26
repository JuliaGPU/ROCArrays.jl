### Temporary definitions to support Flux

## interop with AMDGPUnative

function Base.convert(::Type{ROCDeviceArray{T,N,AS.Global}}, a::ROCArray{T,N}) where {T,N}
  ptr = Base.unsafe_convert(Ptr{T}, a.buf)
  ROCDeviceArray{T,N,AS.Global}(a.dims, DevicePtr{T,AS.Global}(ptr+a.offset))
end

Adapt.adapt_storage(::AMDGPUnative.Adaptor, xs::ROCArray{T,N}) where {T,N} =
  convert(ROCDeviceArray{T,N,AS.Global}, xs)



## interop with CPU array

# We don't convert isbits types in `adapt`, since they are already
# considered GPU-compatible.

Adapt.adapt_storage(::Type{<:ROCArray}, xs::AbstractArray) =
  isbits(xs) ? xs : convert(ROCArray, xs)

Adapt.adapt_storage(::Type{<:ROCArray{T}}, xs::AbstractArray{<:Real}) where T <: AbstractFloat =
  isbits(xs) ? xs : convert(ROCArray{T}, xs)

Adapt.adapt_storage(::Type{<:Array}, xs::ROCArray) = convert(Array, xs)


Base.convert(::Type{R}, x) where R<:ROCArray = ROCArray(x)
Base.convert(::Type{R}, x::ROCArray) where R<:ROCArray = x

roc(xs) = Adapt.adapt(ROCArray{Float32}, xs)
allowscalar(x::Bool) = nothing

