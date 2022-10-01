using YAML
using Flux

using YOLOv7: YOLOChain, Node

struct YOLOBlock{Block} end

function parse_yolo(file)
    yolo_yaml = YAML.load_file(file; dicttype=Dict{Symbol,Any})

    # chain = YOLOChain([3], Vector{Any}([[]]))
    root = Node([], [], (YOLOBlock{:input}(), [0], 1, []), 3)
    node_list = [root]
    for node in yolo_yaml[:backbone]
        
        from = node[1]
        if isa(from, Vector)
            from[from .< 0] .= length(node_list) .+ from .+ 1
        else
            if from < 0
                from = length(node_list) + from + 1
            end
            from = [from]
        end

        number = node[2]
        op = node[3]
        args = node[4]

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

function parse_graph(graph::Vector{Node})
    layers = []
    actual_chain = []
    for node in graph
        n = load_node(graph, node.op...)
        push!(actual_chain, n)

        if length(node.children) == 1
            
        else
            n = Split()
        end
        println(n)
    end
end

function load_node(chain, ::YOLOBlock{:input}, from::Vector{Int}, number::Int, args::Vector)
    return ""
end

function load_node(chain, ::YOLOBlock{:Conv}, from::Vector{Int}, number::Int, args::Vector{Int})
    @assert length(args) == 3
    
    # Julia does not allow negative index
    ch_in = chain[from][1].ch_out

    ch_out = args[1]
    filter = args[2]
    stride = args[3]

    if length(filter) == 1
        filter = (filter, filter)
    end

    # push!(chain.channels, ch_out)

    m = Flux.Conv(filter, ch_in => ch_out; stride=stride) 

    # push!(chain.m, m)

    return m
end

function load_node(chain, ::YOLOBlock{:Concat}, from::Vector{Int}, number::Int, args::Vector{Int})
    @assert length(args) == 1
    @assert number == 1

    dim = 4 - args[1]

    return Flux.Parallel(x -> cat(x; dims=dim), map(x -> x.ch_out, chain[from])...)
end

function load_node(chain, ::YOLOBlock{:MP}, from::Vector{Int}, number::Int, args::Vector{Any})
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