using Flux

using YOLOv7: Node

# custom split layer
struct Split{T}
  paths::T
end

Split(paths...) = Split(paths)

Flux.@functor Split

(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

struct YOLOChain
    nodes::Vector{Node}
    components::Vector{Any}
end

Flux.@functor YOLOChain

Base.getindex(c::YOLOChain, i::Int64) = YOLOChain(c.nodes[i], c.components[i])

Base.getindex(c::YOLOChain, i::UnitRange{Int64}) = YOLOChain(c.nodes[i], c.components[i])

(m::YOLOChain)(x::AbstractArray) = _apply_chain(m, x)

function _apply_chain(m::YOLOChain, x::AbstractArray)
    results = []
    # fs = m.nodes[1]
    # cs = m.components[1]
    # push!(results, cs(x))

    for (node, component) in zip(m.nodes, m.components)
        if length(node.parents) == 0
            push!(results, x)
        else
            # println(node)
            # println(component)
            # println(node.op[2])
            # println(size(results[node.op[2][1]]))
            push!(results, component(results[node.op[2]]...))
        end
        # println(length(results))
    end

    return results[end]
end

struct Conv
    conv::Flux.Conv
    bn::Flux.BatchNorm
    act::Function
end

Conv(conv::Flux.Conv, bn::Flux.BatchNorm) = Conv(conv, bn, silu)

Flux.@functor Conv

function (m::Conv)(x::AbstractArray)
    m.act(m.bn(m.conv(x)))
end

struct SPPCSPC
    cv1::Conv
    cv2::Conv
    cv3::Conv
    cv4::Conv
    m::Array{Flux.MaxPool}
    cv5::Conv
    cv6::Conv
    cv7::Conv
end

# CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
function SPPCSPC(c1, c2; n=1, shortcut=false, g=1, e=0.5, k=(5, 9, 13))
        c_ = Int(2 * c2 * e)  # hidden channels
        cv1 = Conv(Flux.Conv((1, 1), c1 => c_; stride=1, pad=Flux.SamePad()), Flux.BatchNorm(c_))
        cv2 = Conv(Flux.Conv((1, 1), c1 => c_; stride=1, pad=Flux.SamePad()), Flux.BatchNorm(c_))
        cv3 = Conv(Flux.Conv((3, 3), c_ => c_; stride=1, pad=Flux.SamePad()), Flux.BatchNorm(c_))
        cv4 = Conv(Flux.Conv((1, 1), c_ => c_; stride=1, pad=Flux.SamePad()), Flux.BatchNorm(c_))
        m = [Flux.MaxPool((x, x); stride=1, pad=x÷2) for x in k]
        cv5 = Conv(Flux.Conv((1, 1), 4 * c_ => c_; stride=1, pad=Flux.SamePad()), Flux.BatchNorm(c_))
        cv6 = Conv(Flux.Conv((3, 3), c_ => c_; stride=1, pad=Flux.SamePad()), Flux.BatchNorm(c_))
        cv7 = Conv(Flux.Conv((1, 1), 2 * c_ => c2; stride=1, pad=Flux.SamePad()), Flux.BatchNorm(c_))

        return SPPCSPC(cv1, cv2, cv3, cv4, m, cv5, cv6, cv7)
end

function (m::SPPCSPC)(x::AbstractArray)
    x1 = m.cv4(m.cv3(m.cv1(x)))
    c1 = cat(x1, [m(x1) for m in m.m]...; dims=3)
    y1 = m.cv6(m.cv5(c1))
    y2 = m.cv2(x)
    return m.cv7(cat(y1, y2; dims=3))
end

Flux.@functor SPPCSPC