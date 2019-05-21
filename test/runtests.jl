using ROCArrays
using HSARuntime
using LinearAlgebra
using Test

const agent = HSARuntime.get_default_agent()
agent_name = HSARuntime.get_name(agent)
agent_isa = HSARuntime.get_first_isa(agent)
@info "Testing using device $agent_name with ISA $agent_isa"

@testset "ROCArrays" begin

@testset "ROCArrays External Libraries" begin
    isdefined(ROCArrays, :rocBLAS) ? include("blas.jl") : @test_skip "rocBLAS"
end

end # @testset "ROCArrays"
