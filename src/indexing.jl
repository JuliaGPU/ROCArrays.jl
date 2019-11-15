import GPUArrays: allowscalar, @allowscalar

function _getindex(xs::ROCArray{T}, i::Integer) where T
  buf = Mem.view(buffer(xs), (i-1)*sizeof(T))
  return Mem.download(T, buf)[1]
end

function _setindex!(xs::ROCArray{T}, v::T, i::Integer) where T
  buf = Mem.view(buffer(xs), (i-1)*sizeof(T))
  Mem.upload!(buf, T[v])
end
