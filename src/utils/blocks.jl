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
    e::Expr
end

function YOLOChain(nodes::Vector{Node}, components::Vector{Any})
    # results::Vector{AbstractArray}
    x = rand(Float32, 160, 160, 3, 1)
    
    nodes_2_gen = []
    size_2_gen = []
    for n in nodes
        if length(n.op[2]) == 1
            push!(nodes_2_gen, [n.op[2]])
            push!(size_2_gen, Array{Int64, length(n.op[2])}(undef, 1))
        else
            push!(nodes_2_gen, n.op[2])
            push!(size_2_gen, Array{Int64, length(n.op[2])}(undef, [1 for _ in 1:length(n.op[2])]...))
        end
    end
    e = _generate_chain(Tuple(components), Tuple(nodes_2_gen), Tuple(size_2_gen), x)

    return YOLOChain(nodes, components, e)
end

Flux.@functor YOLOChain

function Flux.trainable(m::YOLOChain)
    (m.components,)
end

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
# (m::YOLOChain)(x::AbstractArray) = _apply_chain(Tuple(m.components), m.e, x)

@generated function _apply_chain(layers::Tuple{Vararg{<:Any,N}}, e, x) where {N}
    Expr(:block, e...)
end

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
@generated function _apply_chain(layers::Tuple{Vararg{<:Any,N}}, nodes::Tuple{Vararg{<:Any,N}}, x) where {N}
    quote
        var"##312" = (layers[1])(x)
        var"##313" = (layers[2])(var"##312")
        var"##314" = (layers[3])(var"##313")
        var"##315" = (layers[4])(var"##314")
        var"##316" = (layers[5])(var"##315")
        var"##317" = (layers[6])(var"##315")
        var"##318" = (layers[7])(var"##317")
        var"##319" = (layers[8])(var"##318")
        var"##320" = (layers[9])(var"##319")
        var"##321" = (layers[10])(var"##320")
        var"##322" = (layers[11])(var"##321", var"##319", var"##317", var"##316")
        var"##323" = (layers[12])(var"##322")
        var"##324" = (layers[13])(var"##323")
        var"##325" = (layers[14])(var"##324")
        var"##326" = (layers[15])(var"##323")
        var"##327" = (layers[16])(var"##326")
        var"##328" = (layers[17])(var"##327", var"##325")
        var"##329" = (layers[18])(var"##328")
        var"##330" = (layers[19])(var"##328")
        var"##331" = (layers[20])(var"##330")
        var"##332" = (layers[21])(var"##331")
        var"##333" = (layers[22])(var"##332")
        var"##334" = (layers[23])(var"##333")
        var"##335" = (layers[24])(var"##334", var"##332", var"##330", var"##329")
        var"##336" = (layers[25])(var"##335")
        var"##337" = (layers[26])(var"##336")
        var"##338" = (layers[27])(var"##337")
        var"##339" = (layers[28])(var"##336")
        var"##340" = (layers[29])(var"##339")
        var"##341" = (layers[30])(var"##340", var"##338")
        var"##342" = (layers[31])(var"##341")
        var"##343" = (layers[32])(var"##341")
        var"##344" = (layers[33])(var"##343")
        var"##345" = (layers[34])(var"##344")
        var"##346" = (layers[35])(var"##345")
        var"##347" = (layers[36])(var"##346")
        var"##348" = (layers[37])(var"##347", var"##345", var"##343", var"##342")
        var"##349" = (layers[38])(var"##348")
        var"##350" = (layers[39])(var"##349")
        var"##351" = (layers[40])(var"##350")
        var"##352" = (layers[41])(var"##349")
        var"##353" = (layers[42])(var"##352")
        var"##354" = (layers[43])(var"##353", var"##351")
        var"##355" = (layers[44])(var"##354")
        var"##356" = (layers[45])(var"##354")
        var"##357" = (layers[46])(var"##356")
        var"##358" = (layers[47])(var"##357")
        var"##359" = (layers[48])(var"##358")
        var"##360" = (layers[49])(var"##359")
        var"##361" = (layers[50])(var"##360", var"##358", var"##356", var"##355")
        var"##362" = (layers[51])(var"##361")
        var"##363" = (layers[52])(var"##362")
        var"##364" = (layers[53])(var"##363")
        var"##365" = (layers[54])(var"##364")
        var"##366" = (layers[55])(var"##347")
        var"##367" = (layers[56])(var"##366", var"##365")
        var"##368" = (layers[57])(var"##367")
        var"##369" = (layers[58])(var"##367")
        var"##370" = (layers[59])(var"##369")
        var"##371" = (layers[60])(var"##370")
        var"##372" = (layers[61])(var"##371")
        var"##373" = (layers[62])(var"##372")
        var"##374" = (layers[63])(var"##373", var"##372", var"##371", var"##370", var"##369", var"##368")
        var"##375" = (layers[64])(var"##374")
        var"##376" = (layers[65])(var"##375")
        var"##377" = (layers[66])(var"##376")
        var"##378" = (layers[67])(var"##334")
        var"##379" = (layers[68])(var"##378", var"##377")
        var"##380" = (layers[69])(var"##379")
        var"##381" = (layers[70])(var"##379")
        var"##382" = (layers[71])(var"##381")
        var"##383" = (layers[72])(var"##382")
        var"##384" = (layers[73])(var"##383")
        var"##385" = (layers[74])(var"##384")
        var"##386" = (layers[75])(var"##385", var"##384", var"##383", var"##382", var"##381", var"##380")
        var"##387" = (layers[76])(var"##386")
        var"##388" = (layers[77])(var"##387")
        var"##389" = (layers[78])(var"##388")
        var"##390" = (layers[79])(var"##387")
        var"##391" = (layers[80])(var"##390")
        var"##392" = (layers[81])(var"##391", var"##389", var"##373")
        var"##393" = (layers[82])(var"##392")
        var"##394" = (layers[83])(var"##392")
        var"##395" = (layers[84])(var"##394")
        var"##396" = (layers[85])(var"##395")
        var"##397" = (layers[86])(var"##396")
        var"##398" = (layers[87])(var"##397")
        var"##399" = (layers[88])(var"##398", var"##397", var"##396", var"##395", var"##394", var"##393")
        var"##400" = (layers[89])(var"##399")
        var"##401" = (layers[90])(var"##400")
        var"##402" = (layers[91])(var"##401")
        var"##403" = (layers[92])(var"##400")
        var"##404" = (layers[93])(var"##403")
        var"##405" = (layers[94])(var"##404", var"##402", var"##361")
        var"##406" = (layers[95])(var"##405")
        var"##407" = (layers[96])(var"##405")
        var"##408" = (layers[97])(var"##407")
        var"##409" = (layers[98])(var"##408")
        var"##410" = (layers[99])(var"##409")
        var"##411" = (layers[100])(var"##410")
        var"##412" = (layers[101])(var"##411", var"##410", var"##409", var"##408", var"##407", var"##406")
        var"##413" = (layers[102])(var"##412")
        var"##414" = (layers[103])(var"##385")
        var"##415" = (layers[104])(var"##398")
        var"##416" = (layers[105])(var"##411")
        var"##417" = (layers[106])(var"##416")
        var"##418" = (layers[107])(var"##417")
        var"##419" = (layers[108])(var"##418")
    end
end

function _generate_chain(layers::Tuple{Vararg{<:Any,N}}, nodes::Tuple{Vararg{<:Array{Int64, 1},N}}, sizes::Tuple{Vararg{<:Array{Int64, M},N}}, x) where {N, M}
    symbols = vcat(:x, [gensym() for _ in 1:N])
    # froms = [n.op[2] for n in nodes]
    froms = [gensym() for _ in 1:N]
   
    calls = []
    for i in 1:N
        if M == 1
            q = :($(symbols[i+1]) = layers[$i]($(symbols[(nodes[i])]...)))
            push!(calls, q)
        else
            q = :($(symbols[i+1]) = cat(([symbols$[(nodes[i][j]) for j in 1:M]]...); dims=3) )
            push!(calls, q)
        end
    end
    
    # println(calls)
    # println(symbols,"\n",aux_symbols,"\n",calls)
    # eval(:(x = $x))
    # eval(:(layers = $layers))
    
    # for e in calls
    #     # println(e)
    #     # eval(e)
    # end
    return Expr(:block, calls...)
    # return x
end

# function _apply_chain(layers::Tuple{Vararg{<:Any,N}}, nodes::Tuple{Vararg{<:Any,N}}, x) where {N}
#     # println(m)
#     symbols = vcat(:x, [gensym() for _ in 1:N])
#     froms = [:((nodes[$i].op[2])) for i in 1:N]
#     println("after froms")
#     # symbols = [:x]
#     aux_symbols = []
#     for i in 1:N
#         if length(froms[i]) > 1
#             push!(aux_symbols, [gensym() for _ in 1:length(froms[i])]...)
#         end
#     end
    
#     eval(:(symbols = $symbols))
#     eval(:(aux_symbols = $aux_symbols))

#     # calls_from = [:($(froms[i]) = ((nodes[$i]).op[2])) for i in 1:N]
#     # println(Expr(:block, calls_from...))
#     # calls = [:($(symbols[i+1]) = layers[$i]((symbols[$(nodes[i])]))) for i in 1:N]
#     # println(nodes[1].op[2])
#     # calls = [:($(symbols[i+1]) = length($f) == 1 ? layers[$i]($(symbols[(f)])) : layers[$i](($symbols[$(g for g in f)]))) for (i, f) in zip(1:N, froms)]
#     # calls = [:($(symbols[i+1]) = length($(froms[i])) == 1 ? layers[$i]($(symbols[(froms[i])])) : layers[$i]($(symbols[froms[i]]))) for i in 1:N]
#     calls = []
#     k = 1
#     for i in 1:N
#         if length(froms[i]) == 1
#             q = :($(symbols[i+1]) = layers[$i]($(symbols[(froms[i])])))
#             push!(calls, q)
#         else
#             M = length(froms[i])
#             q = :($(aux_symbols[k]) = $(symbols[(froms[i][1])]))
#             push!(calls, q)
#             k += 1
#             for j in 2:M
#                 :(k = 1)
#                 q = :($(aux_symbols[k]) = cat(($(aux_symbols[k-1]), $(symbols[(froms[i][j])]))...; dims=3))
#                 push!(calls, q)
#                 k += 1
#             end
#             q = :($(symbols[i+1]) = $(aux_symbols[k-1]))
#             push!(calls, q)
#         end
#     end
    
#     # println(calls)
#     # println(symbols,"\n",aux_symbols,"\n",calls)
#     eval(:(x = $x))
#     eval(:(layers = $layers))
    
#     for e in calls
#         # println(e)
#         # eval(e)
#     end
#     Expr(:block, calls...)
#     # return x
# end

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

function (m::Conv)(x::Dict)
    # println(x)
    x[:x] = m.act(m.bn(m.conv(x[:x])))
    return x
end

Base.show(io::IO, obj::Conv) = show(io, "$(obj.conv); $(obj.bn); $(obj.act)")

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
        cv7 = Conv(Flux.Conv((1, 1), 2 * c_ => c2; stride=1, pad=Flux.SamePad()), Flux.BatchNorm(c2))

        return SPPCSPC(cv1, cv2, cv3, cv4, m, cv5, cv6, cv7)
end

function (m::SPPCSPC)(x::AbstractArray)
    x1 = m.cv4(m.cv3(m.cv1(x)))
    c1 = cat(x1, [mp(x1) for mp in m.m]...; dims=3)
    y1 = m.cv6(m.cv5(c1))
    y2 = m.cv2(x)
    return m.cv7(cat(y1, y2; dims=3))
end

function (m::SPPCSPC)(x::Dict)
    ret = m(x[:x])
    x[:x] = ret
    x[:sppcspc] = x[:x]
    
    return x
end

Flux.@functor SPPCSPC

struct RepConv
    rbr_identity::Union{Nothing, BatchNorm}
    rbr_dense::Chain
    rbr_1x1::Chain
    act::Function
end

function RepConv(c1, c2; k=3, s=1, p=nothing, g=1, act=true)
    # groups = g
    # ch_in = c1
    # ch_out = c2

    activation = act ? silu : identity

    rbr_identity = c1 == c2 && s == 1 ? BatchNorm(c1) : nothing
    rbr_dense = Chain(
        Flux.Conv((k, k), c1 => c2; stride=s, pad=SamePad(), groups=g, bias=false),
        BatchNorm(c2)
    )
    rbr_1x1 = Chain(
        Flux.Conv((1, 1), c1 => c2; stride=s, pad=0, groups=g, bias=false),
        BatchNorm(c2)
    )

    return RepConv(rbr_identity, rbr_dense, rbr_1x1, activation)
end

function (m::RepConv)(x::AbstractArray)
    id_out = 0
    if m.rbr_identity !== nothing
        id_out = m.rbr_identity(x)
    end

    println(size(m.rbr_dense(x)))
    println(size(m.rbr_1x1(x)))
    println(size(id_out))

    return m.act(m.rbr_dense(x) .+ m.rbr_1x1(x) .+ id_out)
end

Flux.@functor RepConv

struct YOLOv7BackboneBlock
    depth::Int
    mp::MaxPool
    c1::Conv
    c2::Conv
    c3::Conv
    c4::Conv
    c5::Conv
    c6::Conv
    c7::Conv
    c8::Conv
    c9::Conv
    c10::Conv
end

function YOLOv7BackboneBlock(depth::Int64; half_cut=false, start_mp=true)
    if half_cut
        half_depth = depth ÷ 2
    else
        half_depth = depth
    end

    if start_mp
        mp = MaxPool((2, 2))
    else
        mp = MaxPool((1, 1))
    end

    c1 = Conv(2*depth => depth, 1, 1)
    c2 = Conv(2*depth => depth, 1, 1)
    c3 = Conv(depth => depth, 3, 2)
    # cat1
    c4 = Conv(2*depth => half_depth, 1, 1)
    c5 = Conv(2*depth => half_depth, 1, 1)
    c6 = Conv(half_depth => half_depth, 3, 1)
    c7 = Conv(half_depth => half_depth, 3, 1)
    c8 = Conv(half_depth => half_depth, 3, 1)
    c9 = Conv(half_depth => half_depth, 3, 1)
    # cat2
    c10 = Conv(4*half_depth => 4*half_depth, 1, 1)

    return YOLOv7BackboneBlock(depth, mp, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10)
end

function(m::YOLOv7BackboneBlock)(x::AbstractArray)
    xc1 = m.c1(m.mp(x))
    xc3 = m.c3(m.c2(x))
    cat1 = cat(xc1, xc3; dims=3)
    xc4 = m.c4(cat1)
    xc5 = m.c5(cat1)
    xc7 = m.c7(m.c6(xc5))
    xc9 = m.c9(m.c8(xc7))
    cat2 = cat(xc4, xc5, xc7, xc9; dims=3)
    xc10 = m.c10(cat2)
    
    return xc10
end

Flux.@functor YOLOv7BackboneBlock (mp, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10,)

struct YOLOv7BackboneInit
    depth::Int
    c1::Conv
    c2::Conv
    c3::Conv
    c4::Conv
    c5::Conv
    c6::Conv
    c7::Conv
    c8::Conv
end

function YOLOv7BackboneInit(depth::Int64)
    c1 = Conv(depth => 2*depth, 3, 2)
    c2 = Conv(2*depth => depth, 1, 1)
    c3 = Conv(2*depth => depth, 1, 1)
    c4 = Conv(depth => depth, 3, 1)
    c5 = Conv(depth => depth, 3, 1)
    c6 = Conv(depth => depth, 3, 1)
    c7 = Conv(depth => depth, 3, 1)
    c8 = Conv(4*depth => 4*depth, 1, 1)

    return YOLOv7BackboneInit(depth, c1, c2, c3, c4, c5, c6, c7, c8)
end

function(m::YOLOv7BackboneInit)(x::AbstractArray)
    xc1 = m.c1(x)
    xc2 = m.c2(xc1)
    xc3 = m.c3(xc1)
    xc5 = m.c5(m.c4(xc3))
    xc7 = m.c7(m.c6(xc5))
    cat1 = cat(xc2, xc3, xc5, xc7; dims=3)
    xc8 = m.c8(cat1)
    
    return xc8
end

Flux.@functor YOLOv7BackboneInit (c1, c2, c3, c4, c5, c6, c7, c8,)

struct YOLOv7Backbone
    c1::Conv
    c2::Conv
    c3::Conv
    ybi::YOLOv7BackboneInit
    ybb1::YOLOv7BackboneBlock
    ybb2::YOLOv7BackboneBlock
    ybb3::YOLOv7BackboneBlock
    p3::Bool
    p4::Bool
end

function YOLOv7Backbone(; mps::Array{Bool, 1}=[true, true, true], p3=false, p4=false)
    c1 = Conv(3 => 32, 3, 1)
    c2 = Conv(32 => 64, 3, 2)
    c3 = Conv(64 => 64, 3, 1)

    ybi = YOLOv7BackboneInit(64)
    ybb1 = YOLOv7BackboneBlock(128; start_mp=mps[1])
    ybb2 = YOLOv7BackboneBlock(256; start_mp=mps[2])
    ybb3 = YOLOv7BackboneBlock(512; half_cut=true,  start_mp=mps[3])

    return YOLOv7Backbone(c1, c2, c3, ybi, ybb1, ybb2, ybb3, p3, p4)
end

function(m::YOLOv7Backbone)(x::AbstractArray)
    xp3 = m.ybb1(m.ybi(m.c3(m.c2(m.c1(x)))))
    xp4 = m.ybb2(xp3)
    ret = m.ybb3(xp4)

    ret = Dict(:x=>ret)

    if m.p3
        ret[:p3] = xp3
    end
    if m.p4
        ret[:p4] = xp4
    end

    return ret
end

Flux.@functor YOLOv7Backbone (c1, c2, c3, ybi, ybb1, ybb2, ybb3,)

struct YOLOv7HeadRouteback
    depth::Int
    c1::Conv
    up::Upsample
    cback::Conv
    routeback::Symbol
end

function YOLOv7HeadRouteback(depth::Int, routeback::Symbol)
    c1 = Conv(depth => depth ÷ 2, 1, 1)
    up = Upsample(2, :nearest)
    cback = Conv(2*depth => depth ÷ 2, 1, 1)

    return YOLOv7HeadRouteback(depth, c1, up, cback, routeback)
end

function(m::YOLOv7HeadRouteback)(x::Dict)
    xup = m.up(m.c1(x[:x]))
    xrb = m.cback(x[m.routeback])

    x[:x] = cat(xup, xrb; dims=3)
    return x
end

Flux.@functor YOLOv7HeadRouteback (c1, up, cback,)

struct YOLOv7HeadBlock
    depth::Int
    name::Symbol
    c1::Conv
    c2::Conv
    c3::Conv
    c4::Conv
    c5::Conv
    c6::Conv
    c7::Conv
end

function YOLOv7HeadBlock(depth::Int64, name::Symbol) # 256
    half_depth = depth ÷ 2

    c1 = Conv(2*depth => depth, 1, 1)
    c2 = Conv(2*depth => depth, 1, 1)
    c3 = Conv(depth => half_depth, 3, 1)
    c4 = Conv(half_depth => half_depth, 3, 1)
    c5 = Conv(half_depth => half_depth, 3, 1)
    c6 = Conv(half_depth => half_depth, 3, 1)
    # cat1
    c7 = Conv(4*half_depth + 2*depth => depth, 1, 1)

    return YOLOv7HeadBlock(depth, name, c1, c2, c3, c4, c5, c6, c7)
end

function(m::YOLOv7HeadBlock)(x::AbstractArray)
    xc1 = m.c1(x)
    xc2 = m.c2(x)
    xc3 = m.c3(xc2)
    xc4 = m.c4(xc3)
    xc5 = m.c5(xc4)
    xc6 = m.c6(xc5)
    cat1 = cat(xc1, xc2, xc3, xc4, xc5, xc6; dims=3)
    xc7 = m.c7(cat1)
    
    return xc7
end

function(m::YOLOv7HeadBlock)(x::Dict)
    ret = m(x[:x])

    x[:x] = ret
    x[m.name] = ret
    
    # Both original x and ret have a :x key
    # For the merge function, elements with the same key, 
    # the value for that key will be of the last Dict listed (ret)
    # return merge(x, ret)
    return x
end

Flux.@functor YOLOv7HeadBlock (c1, c2, c3, c4, c5, c6, c7,)

Flux.flatten(x::Dict) = Flux.flatten(x[:x])

struct YOLOv7HeadIncep
    depth::Int
    mp::MaxPool
    c1::Conv
    c2::Conv
    c3::Conv
    routeback::Symbol
end

function YOLOv7HeadIncep(depth::Int, routeback::Symbol)
    mp = MaxPool((2, 2))
    c1 = Conv(depth => depth, 1, 1)
    c2 = Conv(depth => depth, 1, 1)
    c3 = Conv(depth => depth, 3, 2)

    return YOLOv7HeadIncep(depth, mp, c1, c2, c3, routeback)
end

function(m::YOLOv7HeadIncep)(x::Dict)
    xc1 = m.c1(m.mp(x[:x]))
    xc2 = m.c2(x[:x])
    xc3 = m.c3(xc2)

    x[:x] = cat(xc1, xc3, x[m.routeback]; dims=3)
    return x
end

Flux.@functor YOLOv7HeadIncep (mp, c1, c2, c3,)

struct YOLOv7HeadTailRepConv
    depth::Int
    repc1::RepConv
    repc2::RepConv
    repc3::RepConv
    routeback1::Symbol
    routeback2::Symbol
    routeback3::Symbol
end

function YOLOv7HeadTailRepConv(depth::Int, routeback1::Symbol, routeback2::Symbol, routeback3::Symbol)
    repc1 = RepConv(depth, 2*depth, k=3, s=1)
    repc2 = RepConv(2*depth, 4*depth, k=3, s=1)
    repc3 = RepConv(4*depth, 8*depth, k=3, s=1)

    return YOLOv7HeadTailRepConv(depth, repc1, repc2, repc3, routeback1, routeback2, routeback3)
end

function(m::YOLOv7HeadTailRepConv)(x::Dict)
    xrepc1 = m.repc1(x[m.routeback1])
    xrepc2 = m.repc2(x[m.routeback2])
    xrepc3 = m.repc3(x[m.routeback3])

    return xrepc1, xrepc2, xrepc3
end

Flux.@functor YOLOv7HeadTailRepConv (repc1, repc2, repc3,)