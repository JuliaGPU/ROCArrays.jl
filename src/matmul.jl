using LinearAlgebra


function generic_matmatmul!(C::AbstractVecOrMat{R}, A::AbstractVecOrMat{T}, B::AbstractVecOrMat{S}) where {T,S,R}
    if size(A,2) != size(B,1)
        throw(DimensionMismatch("matrix A has dimensions $(size(A)), matrix B has dimensions $(size(B))"))
    end
    if size(C,1) != size(A,1) || size(C,2) != size(B,2)
        throw(DimensionMismatch("result C has dimensions $(size(C)), needs $((size(A,1),size(B,2)))"))
    end
    if isempty(A) || isempty(B)
        return fill!(C, zero(R))
    end

    function kernel(C, A, B)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        j = (blockIdx().y-1) * blockDim().y + threadIdx().y

        if i <= size(A,1) && j <= size(B,2)
            z2 = zero(A[i, 1]*B[1, j] + A[i, 1]*B[1, j])
            Ctmp = convert(promote_type(R, typeof(z2)), z2)
            for k in 1:size(A,2)
                Ctmp += A[i, k]*B[k, j]
            end
            C[i,j] = Ctmp
        end

        return
    end

    max_threads = 256
    threads_x = min(max_threads, size(C,1))
    threads_y = min(max_threads ÷ threads_x, size(C,2))
    group = (threads_x, threads_y)
    grid = ceil.(Int, (size(C,1), size(C,2)) ./ group)

    @roc gridsize=grid groupsize=group kernel(C, A, B)

    C
end

LinearAlgebra.mul!(C::ROCVecOrMat, A::ROCVecOrMat, B::ROCVecOrMat) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::ROCVecOrMat, A::ROCVecOrMat, B::LinearAlgebra.Adjoint{<:Any, <:ROCVecOrMat}) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::ROCVecOrMat, A::ROCVecOrMat, B::LinearAlgebra.Transpose{<:Any, <:ROCVecOrMat}) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::ROCVecOrMat, A::LinearAlgebra.Adjoint{<:Any, <:ROCVecOrMat}, B::ROCVecOrMat) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::ROCVecOrMat, A::LinearAlgebra.Transpose{<:Any, <:ROCVecOrMat}, B::ROCVecOrMat) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::ROCVecOrMat, A::LinearAlgebra.Transpose{<:Any, <:ROCVecOrMat}, B::LinearAlgebra.Adjoint{<:Any, <:ROCVecOrMat}) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::ROCVecOrMat, A::LinearAlgebra.Adjoint{<:Any, <:ROCVecOrMat}, B::LinearAlgebra.Transpose{<:Any, <:ROCVecOrMat}) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::ROCVecOrMat, A::LinearAlgebra.Adjoint{<:Any, <:ROCVecOrMat}, B::LinearAlgebra.Adjoint{<:Any, <:ROCVecOrMat}) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::ROCVecOrMat, A::LinearAlgebra.Transpose{<:Any, <:ROCVecOrMat}, B::LinearAlgebra.Transpose{<:Any, <:ROCVecOrMat}) = generic_matmatmul!(C, A, B)


function generic_rmul!(X::ROCArray, s::Number)
    function kernel(X, s)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        @inbounds X[i] *= s
        return
    end
    @roc groupsize=length(X) gridsize=length(X) kernel(X, s)
    X
end

LinearAlgebra.rmul!(A::ROCArray, b::Number) = generic_rmul!(A, b)


function generic_lmul!(s::Number, X::ROCArray)
    function kernel(s, X)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        @inbounds X[i] = s*X[i]
        return
    end
    @roc groupsize=length(X) gridsize=length(X) kernel(s, X)
    X
end

LinearAlgebra.lmul!(a::Number, B::ROCArray) = generic_lmul!(a, B)
