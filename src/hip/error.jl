export HIPError

struct HIPError <: Exception
    code::hipError_t
    msg::AbstractString
end
Base.show(io::IO, err::HIPError) = print(io, "HIPError(code $(err.code), $(err.msg))")

function HIPError(code::hipError_t)
    msg = status_message(code)
    return HIPError(code, msg)
end

function status_message(status)
    if status == hipSuccess
        return "the operation completed successfully"
    elseif status == hipErrorOutOfMemory
        return "out of memory"
    elseif status == hipErrorNotInitialized
        return "not initialized"
    elseif status == hipErrorDeinitialized
        return "already deinitialized"
    elseif status == hipErrorInvalidHandle
        return "invalid handle"
    elseif status == hipAssert
        return "kernel calls produced assert"
    elseif status == hipErrorNotSupported
        return "hip API is not supported/implemented"
    elseif status == hipErrorNotReady
        return "asynchronous operations not completed"
    else
        return "unknown error"
    end
end


macro check(hip_func)
    quote
        local err::hipError_t
        err = $(esc(hip_func::Expr))
        if err != hipSuccess && err != hipErrorNotReady
            throw(HIPError(err))
        end
        err
    end
end
