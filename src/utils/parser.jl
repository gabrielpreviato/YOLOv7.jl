using YAML
using Flux

# using YOLOv7: Node, Split, YOLOChain, SPPCSPC, silu, YOLOBlock

using YOLOv7

function parse_yolo(file)
    yolo_yaml = YAML.load_file(file; dicttype=Dict{Symbol,Any})

    # chain = YOLOChain([3], Vector{Any}([[]]))
    root = Node([], [], (YOLOBlock{:input}(), [], 1, []), 3, nothing, 0)
    node_list = [root]
    for node in [yolo_yaml[:backbone]; yolo_yaml[:head]]
        
        from = node[1]
        if isa(from, Vector)
            from[from .< 0] .= length(node_list) .+ from[from .< 0] .+ 1
        else
            if from < 0
                from = length(node_list) + from + 1
            end
            from = [from]
        end

        if length(from) == 1
            from = from[1]
        end

        number = node[2]
        op = node[3]
        args = node[4]

        println(node)
        n = Node([], node_list[from], (YOLOBlock{Symbol(op)}(), from, number, args))
        push!(node_list, n)

        for f in from
            push!(node_list[f].children, n)
        end
        
        # m = load_node!(chain, YOLOBlock{Symbol(node[3])}(), from, node[2], node[4])
        # push!(chain.layers, m)
    end
    
    return node_list
end

function create_chain(layer_tuple::Tuple{Vector{Any}, String})
    layers, chain_type = layer_tuple
    if chain_type == "chain"
        return Flux.Chain(create_chain.(layers))
    elseif chain_type == "parallel"
        return Flux.Parallel((x...) -> cat(x...; dims=3), create_chain.(layers)...)
    elseif chain_type == "function"
        return layers[1]
    end
end

function create_chain(layer_tuple::Tuple{Function, String})
    layers, chain_type = layer_tuple
    println(layers)
    println("/////")
    return layers
    # if chain_type == "identity"
    #     return Flux.Chain(create_chain.(layers))
    # elseif chain_type == "parallel"
    #     return Flux.Parallel(create_chain.(layers)...)
    # end
end

function create_chain(layer_tuple::Nothing)
    return
end

function create_chain(layer_tuple::Array)
    println("call")
    return create_chain.(layer_tuple)
end

function create_chain(layer_tuple::Union{Conv, MaxPool, Function})
    return layer_tuple
end

load_node(chain, node::Node) = load_node(chain, node.op...)

function load_node(chain, ::YOLOBlock{:input}, from::Vector{Any}, number::Int, args::Vector)
    # return chain[1]
    return nothing
end

function load_node(chain, ::YOLOBlock{:Conv}, from::Int, number::Int, args::Vector{Int})
    if length(args) <= 3
        activation = silu
    elseif length(args) == 4
        activation = args[4]
    end
    # Julia does not allow negative index
    ch_in = chain[from].ch_out

    ch_out = args[1]
    filter = args[2]
    stride = args[3]

    if length(filter) == 1
        filter = (filter, filter)
    end

    # push!(chain.channels, ch_out)

    conv = Flux.Conv(filter, ch_in => ch_out; stride=stride, pad=Flux.SamePad())
    bn = Flux.BatchNorm(ch_out)
    act = activation

    m = YOLOv7.Conv(conv, bn, act)

    # push!(chain.m, m)

    return m
end

function load_node(chain, ::YOLOBlock{:RepConv}, from::Int, number::Int, args::Vector{Int})
    ch_in = chain[from].ch_out

    ch_out = args[1]
    filter = args[2]
    stride = args[3]

    if length(filter) == 1
        filter = (filter, filter)
    end

    m = YOLOv7.RepConv(ch_in, ch_out; k=filter, s=stride)

    return m
end

function load_node(chain, ::YOLOBlock{:Concat}, from::Vector{Int}, number::Int, args::Vector{Int})
    @assert length(args) == 1
    @assert number == 1

    dim = 4 - args[1]

    # m = Flux.Parallel((x...) -> cat(x...; dims=dim), map(x -> x.node, chain[from])...)
    # println(m)

    return (x...) -> cat(x...; dims=dim)
end

function load_node(chain, ::YOLOBlock{:MP}, from::Int, number::Int, args::Vector{Any})
    if length(args) == 0
        window = (2, 2)
    end

    ch_in = chain[from]
    ch_out = ch_in
    
    # push!(chain.channels, ch_out)

    m = Flux.MaxPool(window)
    # push!(chain.m, m)

    return m
end

function load_node(chain, ::YOLOBlock{:SPPCSPC}, from::Int, number::Int, args::Vector{Int64})
    ch_in = chain[from].ch_out
    ch_out = args[1]
    
    m = SPPCSPC(ch_in, ch_out)

    return m
end

function load_node(chain, ::YOLOBlock{:Flatten}, from::Int, number::Int, args::Vector{Any})
    # ch_in = chain[from][1].ch_out
    # ch_out = args[1]
    
    m = Flux.flatten

    return m
end

function load_node(chain, ::YOLOBlock{:Upsample}, from::Int, number::Int, args::Vector{Any})
    scale = args[2]
    mode = Symbol(args[3])
    
    m = Flux.Upsample(scale, mode)

    return m
end

function load_node(chain, ::YOLOBlock{:Dense}, from::Int, number::Int, args::Union{Vector{Any}, Vector{Int}})
    ch_in = args[1]
    ch_out = args[2]

    if length(args) >= 3
        activation = args[3]
        if activation == "sigmoid"
            activation = Flux.sigmoid
        elseif activation == "relu"
            activation = Flux.relu
        elseif activation == "norm_sigmoid"
            activation = norm_sigmoid
        elseif activation == "softsign"
            activation = NNlib.softsign
        end
    else
        activation = identity
    end
    
    m = Flux.Dense(ch_in => ch_out, activation)

    return m
end

function load_model(file)
    l = parse_yolo(file)
    b = []
    for i in l
        push!(b, load_node(l, i.op...))
    end
    println("Loading model:")
    m = YOLOChain(l[2:end], b[2:end])
    println("Model loaded!")

    return m
end


