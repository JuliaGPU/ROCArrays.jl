using ROCArrays: @rocindex, rocdims

# HACK FIXME: We don't have shmem in AMDGPUnative yet...
macro rocStaticSharedMem(T, len)
    return Vector{T}(undef, len)
end

function mapreducedim_kernel_serial(f, op, R, A, range)
    I = @rocindex R
    newrange = map((r, i) -> r === nothing ? i : r, range, I)
    for I′ in CartesianIndices(newrange)
        @inbounds R[I...] = op(R[I...], f(A[I′]))
    end
    return
end

@inline function reduce_block(arr::ROCDeviceArray, op)
    sync_workgroup()
    len = blockDim().x
    while len != 1
        sync_workgroup()
        skip = (len + 1) >> 1
        reduce_to = threadIdx().x - skip
        if 0 < reduce_to <= (len >> 1)
            arr[reduce_to] = op(arr[reduce_to], arr[threadIdx().x])
        end
        len = skip
    end
    sync_workgroup()
end

function mapreducedim_kernel_parallel(f, op, R::ROCDeviceArray{T}, A::ROCDeviceArray{T},
                             CIS, Rlength, Slength) where {T}
    for Ri_base in 0:(gridDim().x * blockDim().y):(Rlength-1)
        Ri = Ri_base + (blockIdx().x - 1) * blockDim().y + threadIdx().y
        Ri > Rlength && return
        RI = Tuple(CartesianIndices(R)[Ri])
        S = @rocStaticSharedMem(T, 512)
        Si_folded_base = (threadIdx().y - 1) * blockDim().x
        Si_folded = Si_folded_base + threadIdx().x
        # serial reduction of A into S by Slength ÷ xthreads
        for Si_base in 0:blockDim().x:(Slength-1)
            Si = Si_base + threadIdx().x
            Si > Slength && break
            SI = Tuple(CIS[Si])
            AI = ifelse.(size(R) .== 1, SI, RI)
            if Si_base == 0
                S[Si_folded] = f(A[AI...])
            else
                S[Si_folded] = op(S[Si_folded], f(A[AI...]))
            end
        end
        # block-parallel reduction of S to S[1] by xthreads
        reduce_block(view(S, (Si_folded_base + 1):512), op)
        # reduce S[1] into R
        threadIdx().x == 1 && (R[Ri] = op(R[Ri], S[Si_folded]))
    end
    return
end

function Base._mapreducedim!(f, op, R::ROCArray{T}, A::ROCArray{T}) where {T}
    # the kernel as generated from `f` and `op` can require lots of registers (eg. #160),
    # so we need to be careful about how many threads we launch not to run out of them.
    Rlength = length(R)
    Ssize = ifelse.(size(R) .== 1, size(A), 1)
    Slength = prod(Ssize)
    CIS = CartesianIndices(Ssize)

    parallel_args = (f, op, R, A, CIS, Rlength, Slength)
    GC.@preserve parallel_args begin
        parallel_kargs = rocconvert.(parallel_args)
        parallel_tt = Tuple{Core.Typeof.(parallel_kargs)...}
        parallel_kernel = rocfunction(mapreducedim_kernel_parallel, parallel_tt)

        # we are limited in how many threads we can launch...
        ## by the kernel
        kernel_threads = AMDGPUnative.maxthreads(parallel_kernel)
        ## by the device
        dev = HSARuntime.device()
        block_threads = (x=attribute(dev, HSARuntime.MAX_BLOCK_DIM_X),
                         y=attribute(dev, HSARuntime.MAX_BLOCK_DIM_Y),
                         total=attribute(dev, HSARuntime.MAX_THREADS_PER_BLOCK))

        # figure out a legal launch configuration
        y_thr = min(nextpow(2, Rlength ÷ 512 + 1), 512, block_threads.y, kernel_threads)
        x_thr = min(512 ÷ y_thr, Slength, block_threads.x,
                    ceil(Int, block_threads.total/y_thr),
                    ceil(Int, kernel_threads/y_thr))

        if x_thr >= 8
            blk, thr = (Rlength - 1) ÷ y_thr + 1, (x_thr, y_thr, 1)
            parallel_kernel(parallel_kargs...; threads=thr, blocks=blk)
        else
            # not enough work, fall back to serial reduction
            range = ifelse.(length.(axes(R)) .== 1, axes(A), nothing)
            blk, thr = rocdims(R)
            @roc(gridsize=blk, groupsize=thr, mapreducedim_kernel_serial(f, op, R, A, range))
        end
    end

    return R
end
