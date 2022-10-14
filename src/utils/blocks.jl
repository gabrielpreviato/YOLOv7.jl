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
@generated function _apply_chain(layers::Tuple{Vararg{<:Any,N}}, nodes::Tuple{Vararg{<:Any,N}}, x) where {N}
    quote
        var"##736" = (layers[1])(x)
        var"##737" = (layers[2])(var"##736")
        var"##738" = (layers[3])(var"##737")
        var"##739" = (layers[4])(var"##738")
        var"##740" = (layers[5])(var"##739")
        var"##741" = (layers[6])(var"##739")
        var"##742" = (layers[7])(var"##741")
        var"##743" = (layers[8])(var"##742")
        var"##744" = (layers[9])(var"##743")
        var"##745" = (layers[10])(var"##744")
        var"##842" = var"##745"
        var"##843" = cat((var"##842", var"##743")...; dims = 3)
        var"##844" = cat((var"##843", var"##741")...; dims = 3)
        var"##845" = cat((var"##844", var"##740")...; dims = 3)
        var"##746" = var"##845"
        var"##747" = (layers[12])(var"##746")
        var"##748" = (layers[13])(var"##747")
        var"##749" = (layers[14])(var"##748")
        var"##750" = (layers[15])(var"##747")
        var"##751" = (layers[16])(var"##750")
        var"##846" = var"##751"
        var"##847" = cat((var"##846", var"##749")...; dims = 3)
        var"##752" = var"##847"
        var"##753" = (layers[18])(var"##752")
        var"##754" = (layers[19])(var"##752")
        var"##755" = (layers[20])(var"##754")
        var"##756" = (layers[21])(var"##755")
        var"##757" = (layers[22])(var"##756")
        var"##758" = (layers[23])(var"##757")
        var"##848" = var"##758"
        var"##849" = cat((var"##848", var"##756")...; dims = 3)
        var"##850" = cat((var"##849", var"##754")...; dims = 3)
        var"##851" = cat((var"##850", var"##753")...; dims = 3)
        var"##759" = var"##851"
        var"##760" = (layers[25])(var"##759")
        var"##761" = (layers[26])(var"##760")
        var"##762" = (layers[27])(var"##761")
        var"##763" = (layers[28])(var"##760")
        var"##764" = (layers[29])(var"##763")
        var"##852" = var"##764"
        var"##853" = cat((var"##852", var"##762")...; dims = 3)
        var"##765" = var"##853"
        var"##766" = (layers[31])(var"##765")
        var"##767" = (layers[32])(var"##765")
        var"##768" = (layers[33])(var"##767")
        var"##769" = (layers[34])(var"##768")
        var"##770" = (layers[35])(var"##769")
        var"##771" = (layers[36])(var"##770")
        var"##854" = var"##771"
        var"##855" = cat((var"##854", var"##769")...; dims = 3)
        var"##856" = cat((var"##855", var"##767")...; dims = 3)
        var"##857" = cat((var"##856", var"##766")...; dims = 3)
        var"##772" = var"##857"
        var"##773" = (layers[38])(var"##772")
        var"##774" = (layers[39])(var"##773")
        var"##775" = (layers[40])(var"##774")
        var"##776" = (layers[41])(var"##773")
        var"##777" = (layers[42])(var"##776")
        var"##858" = var"##777"
        var"##859" = cat((var"##858", var"##775")...; dims = 3)
        var"##778" = var"##859"
        var"##779" = (layers[44])(var"##778")
        var"##780" = (layers[45])(var"##778")
        var"##781" = (layers[46])(var"##780")
        var"##782" = (layers[47])(var"##781")
        var"##783" = (layers[48])(var"##782")
        var"##784" = (layers[49])(var"##783")
        var"##860" = var"##784"
        var"##861" = cat((var"##860", var"##782")...; dims = 3)
        var"##862" = cat((var"##861", var"##780")...; dims = 3)
        var"##863" = cat((var"##862", var"##779")...; dims = 3)
        var"##785" = var"##863"
        var"##786" = (layers[51])(var"##785")
        var"##787" = (layers[52])(var"##786")
        var"##788" = (layers[53])(var"##787")
        var"##789" = (layers[54])(var"##788")
        var"##790" = (layers[55])(var"##771")
        var"##864" = var"##790"
        var"##865" = cat((var"##864", var"##789")...; dims = 3)
        var"##791" = var"##865"
        var"##792" = (layers[57])(var"##791")
        var"##793" = (layers[58])(var"##791")
        var"##794" = (layers[59])(var"##793")
        var"##795" = (layers[60])(var"##794")
        var"##796" = (layers[61])(var"##795")
        var"##797" = (layers[62])(var"##796")
        var"##866" = var"##797"
        var"##867" = cat((var"##866", var"##796")...; dims = 3)
        var"##868" = cat((var"##867", var"##795")...; dims = 3)
        var"##869" = cat((var"##868", var"##794")...; dims = 3)
        var"##870" = cat((var"##869", var"##793")...; dims = 3)
        var"##871" = cat((var"##870", var"##792")...; dims = 3)
        var"##798" = var"##871"
        var"##799" = (layers[64])(var"##798")
        var"##800" = (layers[65])(var"##799")
        var"##801" = (layers[66])(var"##800")
        var"##802" = (layers[67])(var"##758")
        var"##872" = var"##802"
        var"##873" = cat((var"##872", var"##801")...; dims = 3)
        var"##803" = var"##873"
        var"##804" = (layers[69])(var"##803")
        var"##805" = (layers[70])(var"##803")
        var"##806" = (layers[71])(var"##805")
        var"##807" = (layers[72])(var"##806")
        var"##808" = (layers[73])(var"##807")
        var"##809" = (layers[74])(var"##808")
        var"##874" = var"##809"
        var"##875" = cat((var"##874", var"##808")...; dims = 3)
        var"##876" = cat((var"##875", var"##807")...; dims = 3)
        var"##877" = cat((var"##876", var"##806")...; dims = 3)
        var"##878" = cat((var"##877", var"##805")...; dims = 3)
        var"##879" = cat((var"##878", var"##804")...; dims = 3)
        var"##810" = var"##879"
        var"##811" = (layers[76])(var"##810")
        var"##812" = (layers[77])(var"##811")
        var"##813" = (layers[78])(var"##812")
        var"##814" = (layers[79])(var"##811")
        var"##815" = (layers[80])(var"##814")
        var"##880" = var"##815"
        var"##881" = cat((var"##880", var"##813")...; dims = 3)
        var"##882" = cat((var"##881", var"##797")...; dims = 3)
        var"##816" = var"##882"
        var"##817" = (layers[82])(var"##816")
        var"##818" = (layers[83])(var"##816")
        var"##819" = (layers[84])(var"##818")
        var"##820" = (layers[85])(var"##819")
        var"##821" = (layers[86])(var"##820")
        var"##822" = (layers[87])(var"##821")
        var"##883" = var"##822"
        var"##884" = cat((var"##883", var"##821")...; dims = 3)
        var"##885" = cat((var"##884", var"##820")...; dims = 3)
        var"##886" = cat((var"##885", var"##819")...; dims = 3)
        var"##887" = cat((var"##886", var"##818")...; dims = 3)
        var"##888" = cat((var"##887", var"##817")...; dims = 3)
        var"##823" = var"##888"
        var"##824" = (layers[89])(var"##823")
        var"##825" = (layers[90])(var"##824")
        var"##826" = (layers[91])(var"##825")
        var"##827" = (layers[92])(var"##824")
        var"##828" = (layers[93])(var"##827")
        var"##889" = var"##828"
        var"##890" = cat((var"##889", var"##826")...; dims = 3)
        var"##891" = cat((var"##890", var"##785")...; dims = 3)
        var"##829" = var"##891"
        var"##830" = (layers[95])(var"##829")
        var"##831" = (layers[96])(var"##829")
        var"##832" = (layers[97])(var"##831")
        var"##833" = (layers[98])(var"##832")
        var"##834" = (layers[99])(var"##833")
        var"##835" = (layers[100])(var"##834")
        var"##892" = var"##835"
        var"##893" = cat((var"##892", var"##834")...; dims = 3)
        var"##894" = cat((var"##893", var"##833")...; dims = 3)
        var"##895" = cat((var"##894", var"##832")...; dims = 3)
        var"##896" = cat((var"##895", var"##831")...; dims = 3)
        var"##897" = cat((var"##896", var"##830")...; dims = 3)
        var"##836" = var"##897"
        var"##837" = (layers[102])(var"##836")
        var"##838" = (layers[103])(var"##837")
        var"##839" = (layers[104])(var"##838")
        var"##840" = (layers[105])(var"##839")
        var"##841" = (layers[106])(var"##840")
    end
end

function _generate_call(layers::Tuple{Vararg{<:Any,N}}, nodes::Tuple{Vararg{<:Any,N}}, x) where {N}
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
    # calls = [:($(symbols[i+1]) = length($(froms[i])) == 1 ? layers[$i]($(symbols[(froms[i])])) : layers[$i]($(symbols[froms[i]]))) for i in 1:N]
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
    Expr(:block, calls...)
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