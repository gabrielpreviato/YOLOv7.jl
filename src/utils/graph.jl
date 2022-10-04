mutable struct Node
    children::Vector
    parents::Vector
    op::Any
    ch_out::Int
    node::Any
    visited::Int
end

Node() = Node([], [], nothing, 0, nothing, 0)

function Node(children, parents, op)
    args = op[4]
    if length(args) == 3
        ch_out = args[1]
    else
        if length(parents) > 0
            ch_out = sum(x -> x.ch_out, parents)
        else
            ch_out = 0
        end
    end

    return Node(children, parents, op, ch_out, nothing, 0)
end

function Base.show(io::IO, m::Node)
    print(io, m.op, length(m.children))
end