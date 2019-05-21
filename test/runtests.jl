using HSAArrays, HSAArrays.rocBLAS
using HSARuntime
using LinearAlgebra
using Test

const agent = HSARuntime.get_default_agent()
agent_name = HSARuntime.get_name(agent)
agent_isa = HSARuntime.get_first_isa(agent)
@info "Testing using device $agent_name with ISA $agent_isa"

handle = rocBLAS.rocblas_create_handle()

@testset "Level 1 BLAS" begin
    @testset "scal()" begin
        for T in (Float32, Float64)
            A = rand(T, 8)
            HA = HSAArray(agent, A)
            if T === Float32
                rocBLAS.rocblas_sscal(handle, 8, 5f0, HA, 1)
            else
                rocBLAS.rocblas_dscal(handle, 8, 5.0, HA, 1)
            end
            _A = Array(HA)
            @test isapprox(A .* 5, _A)
        end
    end
    @testset "copy()" begin
        for T in (Float32, Float64)
            A = rand(T, 8)
            B = rand(T, 8)
            HA = HSAArray(agent, A)
            HB = HSAArray(agent, B)
            if T === Float32
                rocBLAS.rocblas_scopy(handle, 8, HA, 1, HB, 1)
            else
                rocBLAS.rocblas_dcopy(handle, 8, HA, 1, HB, 1)
            end
            _A = Array(HA)
            _B = Array(HB)
            @test isapprox(A, _A)
            @test isapprox(A, _B)
        end
    end
    @testset "dot()" begin
        for T in (Float32, Float64)
            A = rand(T, 8)
            B = rand(T, 8)
            result = zeros(T, 8)
            HA = HSAArray(agent, A)
            HB = HSAArray(agent, B)
            result = Ref{T}(zero(T))
            if T === Float32
                rocBLAS.rocblas_sdot(handle, 8, HA, 1, HB, 1, result)
            else
                rocBLAS.rocblas_ddot(handle, 8, HA, 1, HB, 1, result)
            end
            @test isapprox(LinearAlgebra.dot(A,B), result[])
        end
    end
    @testset "swap()" begin
        for T in (Float32, Float64)
            A = rand(T, 8)
            B = rand(T, 8)
            result = zeros(T, 8)
            HA = HSAArray(agent, A)
            HB = HSAArray(agent, B)
            if T === Float32
                rocBLAS.rocblas_sswap(handle, 8, HA, 1, HB, 1)
            else
                rocBLAS.rocblas_dswap(handle, 8, HA, 1, HB, 1)
            end
            _A = Array(HA)
            _B = Array(HB)
            @test isapprox(A, _B)
            @test isapprox(B, _A)
        end
    end
end

rocBLAS.rocblas_destroy_handle(handle)
