struct Node
    children::Vector
    parents::Vector
    op::Any
    ch_out::Int
end

Node() = Node([], [], nothing, 0)

function Node(children, parents, op)
    args = op[4]
    if length(args) == 3
        ch_out = args[1]
    else
        ch_out = 0
    end

    return Node(children, parents, op, ch_out)
end

function Base.show(io::IO, m::Node)
    print(io, m.op)
end