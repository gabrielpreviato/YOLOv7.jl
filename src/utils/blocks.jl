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
    # results::Vector{AbstractArray}
end

Flux.@functor YOLOChain

Flux.trainable(m::YOLOChain) = (m.components,)

# function YOLOChain(nodes::Vector{Node}, components::Vector{Any})
#     # results = []
#     # x = randn(Float32, 640, 640, 3, 1)
#     # for (node, component) in zip(nodes, components)
#     #     if length(node.parents) == 0
#     #         push!(results, x)
#     #     else
#     #         # println(node)
#     #         # println(component)
#     #         # println(node.op[2])
#     #         # println(size(results[node.op[2][1]]))
#     #         push!(results, component(results[node.op[2]]...))
#     #     end
#     #     # println(length(results))
#     # end

#     return YOLOChain(nodes, components)
# end

Base.getindex(c::YOLOChain, i::Int64) = YOLOChain(c.nodes[i], c.components[i])

Base.getindex(c::YOLOChain, i::UnitRange{Int64}) = YOLOChain(c.nodes[i], c.components[i])

(m::YOLOChain)(x::AbstractArray) = _apply_chain(Tuple(m.components), Tuple(m.nodes), x)
# (m::YOLOChain)(x::AbstractArray) = _apply_chain(Tuple(m.components), x)

function _chain(m, node, component, x::AbstractArray)
    # println(node)
    if length(node.parents) == 0
        return x
    end

    return component([_chain(m, m.nodes[i], m.components[i], x) for i in node.op[2]]...)
    # return _chain(m, m.nodes[i], m.components[i], component(x))
end

@generated function _apply_chain(layers::Tuple{Vararg{<:Any,N}}, x) where {N}
    symbols = vcat(:x, [gensym() for _ in 1:N])
    calls = [:($(symbols[i+1]) = layers[$i]($(symbols[i]))) for i in 1:N]
    # println(symbols)
    println(symbols,"\n",calls)
    Expr(:block, calls...)
  end

# @eval function _apply_chain(layers::Tuple{Vararg{<:Any,N}}, nodes::Tuple{Vararg{<:Any,N}}, x) where {N}
#     # println(m)
#     symbols = vcat(:x, [gensym() for _ in 1:N])
#     froms = [n.op[2] for n in nodes]

#     # calls_from = [:($(froms[i]) = ((nodes[$i]).op[2])) for i in 1:N]
#     # println(Expr(:block, calls_from...))
#     # calls = [:($(symbols[i+1]) = layers[$i]((symbols[$(nodes[i])]))) for i in 1:N]
#     # println(nodes[1].op[2])
#     calls = [:($(symbols[i+1]) = layers[$i]($(symbols[(f...)]))) for (i, f, s) in zip(1:N, froms, symbols)]
#     # println(calls)
#     println(symbols,"\n",calls)
#     eval(:(x = $x))
#     eval(:(layers = $layers))
#     eval(:(symbols = $symbols))
#     eval.(calls)
# end

function _apply_chain(layers::Tuple{Vararg{<:Any,N}}, nodes::Tuple{Vararg{<:Any,N}}, x) where {N}
    # println(m)
    symbols = vcat(:x, [gensym() for _ in 1:N])
    froms = [n.op[2] for n in nodes]
    # symbols = [:x]
    aux_symbols = []
    for i in 1:N
        if length(froms[i]) > 1
            push!(aux_symbols, [gensym() for _ in 1:length(froms[i])]...)
        end
    end
    
    eval(:(symbols = $symbols))
    eval(:(aux_symbols = $aux_symbols))

    # calls_from = [:($(froms[i]) = ((nodes[$i]).op[2])) for i in 1:N]
    # println(Expr(:block, calls_from...))
    # calls = [:($(symbols[i+1]) = layers[$i]((symbols[$(nodes[i])]))) for i in 1:N]
    # println(nodes[1].op[2])
    # calls = [:($(symbols[i+1]) = length($f) == 1 ? layers[$i]($(symbols[(f)])) : layers[$i](($symbols[$(g for g in f)]))) for (i, f) in zip(1:N, froms)]
    calls = [:($(symbols[i+1]) = length($(froms[i])) == 1 ? layers[$i]($(symbols[(froms[i])])) : layers[$i]($(symbols[froms[i]]))) for i in 1:N]
    calls = []
    k = 1
    for i in 1:N
        if length(froms[i]) == 1
            q = :($(symbols[i+1]) = layers[$i]($(symbols[(froms[i])])))
            push!(calls, q)
        else
            M = length(froms[i])
            q = :($(aux_symbols[k]) = $(symbols[(froms[i][1])]))
            push!(calls, q)
            k += 1
            for j in 2:M
                :(k = 1)
                q = :($(aux_symbols[k]) = cat(($(aux_symbols[k-1]), $(symbols[(froms[i][j])]))...; dims=3))
                push!(calls, q)
                k += 1
            end
            q = :($(symbols[i+1]) = $(aux_symbols[k-1]))
            push!(calls, q)
        end
    end
    
    # println(calls)
    # println(symbols,"\n",aux_symbols,"\n",calls)
    eval(:(x = $x))
    eval(:(layers = $layers))
    
    for e in calls
        # println(e)
        # eval(e)
    end
    eval(Expr(:block, calls...))
    # return x
end

function _applychain(m::YOLOChain, x::AbstractArray)
    # fs = m.nodes[1]
    # cs = m.components[1]
    # push!(results, cs(x))
    return _chain(m, m.nodes[end], m.components[end], x)

    # for (i, (node, component)) in enumerate(zip(m.nodes, m.components))
    #     if length(node.parents) == 0
    #         m.results[i] = x
    #     else
    #         if length(node.parents) != 1 || node.parents[1] != -1
    #             x = m.results[node.op[2]]
    #         end
    #         # println(node)
    #         # println(component)
    #         # println(node.op[2])
    #         # println(size(results[node.op[2][1]]))
    #         x = component(x...)
    #         m.results[i] = x
    #     end
    #     # println(length(results))
    # end

    # return x
end

struct Conv
    conv::Flux.Conv
    bn::Flux.BatchNorm
    act::Function
end

Conv(conv::Flux.Conv, bn::Flux.BatchNorm) = Conv(conv, bn, silu)

function Conv(c::Pair{Int64, Int64}, kernel, stride)
    return Conv(
        Flux.Conv((kernel, kernel), c; stride=stride, pad=Flux.SamePad()),
        Flux.BatchNorm(c.second)
    )
end

Flux.@functor Conv

function (m::Conv)(x::AbstractArray)
    # println(x)
    m.act(m.bn(m.conv(x)))
end

# function (m::Conv)(x::Symbol)
#     # println(x)
#     return x
# end

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