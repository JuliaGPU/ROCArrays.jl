import Base.Broadcast: Broadcasted, Extruded, BroadcastStyle, ArrayStyle

BroadcastStyle(::Type{<:ROCArray}) = ArrayStyle{ROCArray}()

function Base.similar(bc::Broadcasted{ArrayStyle{ROCArray}}, ::Type{T}) where T
    similar(ROCArray{T}, axes(bc))
end


# replace base functions with devicelibs alternatives
# TODO: do this with Cassette.jl

rocfunc(f) = f
rocfunc(::Type{T}) where T = (x...) -> T(x...) # broadcasting type ctors isn't GPU compatible

Broadcast.broadcasted(::ArrayStyle{ROCArray}, f, args...) =
  Broadcasted{ArrayStyle{ROCArray}}(rocfunc(f), args, nothing)

devicelibs = :[
  cos, cospi, sin, sinpi, tan, acos, asin, atan,
  cosh, sinh, tanh, acosh, asinh, atanh,
  log, log10, log1p, log2, logb, ilogb,
  exp, exp2, exp10, expm1, ldexp,
  erf, erfinv, erfc, erfcinv, erfcx,
  brev, clz, ffs, byte_perm, popc,
  isfinite, isinf, isnan, nearbyint,
  nextafter, signbit, copysign, abs,
  sqrt, rsqrt, cbrt, rcbrt, pow,
  ceil, floor, saturate,
  lgamma, tgamma,
  j0, j1, jn, y0, y1, yn,
  normcdf, normcdfinv, hypot,
  fma, sad, dim, mul24, mul64hi, hadd, rhadd, scalbn].args

for f in devicelibs
  isdefined(Base, f) || continue
  @eval rocfunc(::typeof(Base.$f)) = AMDGPUnative.$f
end

using MacroTools

const _rocfuncs = copy(devicelibs)
rocfuncs() = (global _rocfuncs; _rocfuncs)

function replace_device(ex)
  global _rocfuncs
  MacroTools.postwalk(ex) do x
    x in _rocfuncs ? :(ROCArrays.rocfunc($x)) : x
  end
end

macro rocfunc(ex)
  global _rocfuncs
  def = MacroTools.splitdef(ex)
  f = def[:name]
  def[:name] = Symbol(:cu, f)
  def[:body] = replace_device(def[:body])
  push!(_rocfuncs, f)
  quote
    $(esc(MacroTools.combinedef(def)))
    ROCArrays.rocfunc(::typeof($(esc(f)))) = $(esc(def[:name]))
  end
end

# ForwardDiff Integration
using ForwardDiff: Dual, value, partials, unary_dual_definition
using DiffRules

for f in devicelibs
  if haskey(DiffRules.DEFINED_DIFFRULES, (:Base,f,1))
    f == :tanh && continue
    diffrule = DiffRules.DEFINED_DIFFRULES[(:Base,f,1)]
    DiffRules.DEFINED_DIFFRULES[(:AMDGPUnative,f,1)] =
      (args...) -> replace_device(diffrule(args...))
    eval(unary_dual_definition(:AMDGPUnative, f))
  end
end

DiffRules.DEFINED_DIFFRULES[(:AMDGPUnative, :tanh, 1)] = x ->
  replace_device(:(1-tanh(x)^2))
eval(unary_dual_definition(:AMDGPUnative, :tanh))
