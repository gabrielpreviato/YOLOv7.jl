
struct YOLOChain{T<:Union{Tuple, NamedTuple, AbstractVector}}
    layers::T
end