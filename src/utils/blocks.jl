using Flux

# custom split layer
struct Split{T}
  paths::T
end

Split(paths...) = Split(paths)

Flux.@functor Split

(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

struct YOLOChain{T<:AbstractVector}
    channels::T
    m::Vector{Any}
end

function YOLOChain(channels::T) where T<:Union{Tuple, NamedTuple, AbstractVector}
    return YOLOChain(channels, [])
end

function Base.getindex(chain::YOLOChain, index::Int)
    return Base.getindex(chain.channels, index)
end

function Base.getindex(chain::YOLOChain, index::Vector{Int})
    return Base.getindex(chain.channels, index)
end

function Base.length(chain::YOLOChain)
    return Base.length(chain.channels)
end

