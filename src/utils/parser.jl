using YAML
using Flux

using YOLOv7: YOLOChain, Node, Split

struct YOLOBlock{Block} end

function parse_yolo(file)
    yolo_yaml = YAML.load_file(file; dicttype=Dict{Symbol,Any})

    # chain = YOLOChain([3], Vector{Any}([[]]))
    root = Node([], [], (YOLOBlock{:input}(), [0], 1, []), 3, nothing, 0)
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

function parse_node(graph, node::Node)
    node.visited += 1
    to_visit = nothing
    node.node = load_node(graph, node)
    actual_chain::Vector{Any} = []
    if node.node !== nothing
        actual_chain = [node.node]
    end

    if length(node.parents) == 1 || node.op[1] == YOLOBlock{:input}()
        if length(node.children) == 1
            n, to_visit = parse_node(graph, node.children[1])
            push!(actual_chain, n...)
        elseif length(node.children) > 1
            # n = load_node(graph, c.op...)
            children_chain = []
            for c in node.children
                n_child, to_visit = parse_node(graph, c)
                push!(children_chain, n_child)
            end
            u = Parallel((x...) -> cat(x...; dims=3), children_chain...)
            push!(actual_chain, u)

            if to_visit !== nothing
                println("visiring")
                u = parse_node(graph, to_visit)[1]
                push!(actual_chain, u)
            end
        else
        end
    else
        if node.visited == length(node.parents)
            to_visit = node.children[1]
        # u = parse_node(graph, c)
        #     push!(actual_chain, u)
        #     # println(u)
        end
    end

    return Flux.Chain(actual_chain), to_visit
end

function parse_graph(graph::Vector{Node})
    layers = []  
    
    s::Node = graph[1]

    parse_node(graph, s)

    # n = load_node(graph, s.op...)
    # explore = [n]
    # actual_chain = []
    # while !isempty(explore)
    #     u = pop!(explore)
    #     println(u)
    #     push!(actual_chain, u)

    #     if length(u.children) > 1
    #         children_chains, next = parse_children(graph, u)
    #         u = Split(children_chains)
    #         push!(actual_chain, u)
            
    #         push!(layers, actual_chain)
    #         actual_chain = []
    #     else
    #         next = u.children[1]
    #     end

    #     push!(explore, next)
    # end

    # push!(layers, actual_chain)
    # println(layers)
end

load_node(chain, node::Node) = load_node(chain, node.op...)

function load_node(chain, ::YOLOBlock{:input}, from::Vector{Int}, number::Int, args::Vector)
    # return chain[1]
    return nothing
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

    m = Flux.Conv(filter, ch_in => ch_out; stride=stride, pad=Flux.SamePad()) 

    # push!(chain.m, m)

    return m
end

function load_node(chain, ::YOLOBlock{:Concat}, from::Vector{Int}, number::Int, args::Vector{Int})
    @assert length(args) == 1
    @assert number == 1

    dim = 4 - args[1]

    m = Flux.Parallel(x -> cat(x; dims=dim), map(x -> x.node, chain[from])...)
    # println(m)

    return nothing
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

data = parse_graph(parse_yolo(raw"C:\Users\Previato\YOLOv7.jl\src\conf\yolov7.yaml"))
m = Flux.Chain(data[2:end])
println(m)