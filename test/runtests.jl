using ROCArrays
using HSARuntime
using LinearAlgebra
using Test

using Random
Random.seed!(1)

const agent = HSARuntime.get_default_agent()
agent_name = HSARuntime.get_name(agent)
agent_isa = HSARuntime.get_first_isa(agent)
@info "Testing using device $agent_name with ISA $agent_isa"

# GPUArrays has a testsuite that isn't part of the main package.
# Include it directly.
import GPUArrays
gpuarrays = pathof(GPUArrays)
gpuarrays_root = dirname(dirname(gpuarrays))
include(joinpath(gpuarrays_root, "test", "testsuite.jl"))

testf(f, xs...; kwargs...) = GPUArrays.TestSuite.compare(f, ROCArray, xs...; kwargs...)

import ROCArrays: allowscalar, @allowscalar
allowscalar(false)

@testset "ROCArrays" begin

@testset "GPUArrays test suite" begin
    TestSuite.test_construction(ROCArray)
    TestSuite.test_gpuinterface(ROCArray)
    TestSuite.test_indexing(ROCArray)
    TestSuite.test_io(ROCArray)
    TestSuite.test_base(ROCArray)
    TestSuite.test_mapreduce(ROCArray)
    TestSuite.test_broadcasting(ROCArray)
    TestSuite.test_linalg(ROCArray)
    #TestSuite.test_fft(ROCArray)
    #TestSuite.test_blas(ROCArray)
    #TestSuite.test_random(ROCArray)
end

#=
@testset "ROCArrays External Libraries" begin
    isdefined(ROCArrays, :rocBLAS) ? include("blas.jl") : @test_skip "rocBLAS"
    isdefined(ROCArrays, :rocFFT) ? include("fft.jl") : @test_skip "rocFFT"
end
=#

end # @testset "ROCArrays"
