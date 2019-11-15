import Base: view

using Base: ScalarIndex, ViewIndex, Slice, @_inline_meta, @boundscheck, 
            to_indices, compute_offset1, unsafe_length, _maybe_reshape_parent, index_ndims

struct Contiguous end
struct NonContiguous end

# Detect whether the view is contiguous or not
ROCIndexStyle() = Contiguous()
ROCIndexStyle(I...) = NonContiguous()
ROCIndexStyle(i1::Colon, ::ScalarIndex...) = Contiguous()
ROCIndexStyle(i1::AbstractUnitRange, ::ScalarIndex...) = Contiguous()
ROCIndexStyle(i1::Colon, I...) = ROCIndexStyle(I...)

rocviewlength() = ()
rocviewlength(::Real, I...) = (@_inline_meta; rocviewlength(I...)) # skip scalars
rocviewlength(i1::AbstractUnitRange, I...) = (@_inline_meta; (unsafe_length(i1), rocviewlength(I...)...))
rocviewlength(i1::AbstractUnitRange, ::ScalarIndex...) = (@_inline_meta; (unsafe_length(i1),))

view(A::ROCArray, I::Vararg{Any,N}) where {N} = (@_inline_meta; _rocview(A, I, ROCIndexStyle(I...)))

function _rocview(A, I, ::Contiguous)
    @_inline_meta
    J = to_indices(A, I)
    @boundscheck checkbounds(A, J...)
    _rocview(_maybe_reshape_parent(A, index_ndims(J...)), J, rocviewlength(J...))
end

# for contiguous views just return a new ROCArray
_rocview(A::ROCArray{T}, I::NTuple{N,ViewIndex}, dims::NTuple{M,Integer}) where {T,N,M} =
    ROCArray{T,M}(A.buf, dims; offset=A.offset + compute_offset1(A, 1, I) * sizeof(T), own=A.own)

# fallback to SubArray when the view is not contiguous
_rocview(A, I, ::NonContiguous) where {N} = invoke(view, Tuple{AbstractArray, typeof(I).parameters...}, A, I...)
