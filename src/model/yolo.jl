using Flux

struct yolov7
    name::String
    pretrained::Bool
    class_names
    m
end

function fuse(m::yolov7)
    layers = []
    for layer in m.m
        push!(layers, fuse(layer))
    end

    return yolov7(m.name, m.pretrained, m.class_names, Chain(layers))
end

function yolov7(;name="yolov7", nc=1, class_names=["obstacle"], anchor_grid=(), anchors=())
    @assert nc == length(class_names) "Number of classes (nc: $nc) need to match the number of class names ($(length(class_names)))"
    
    anchors, anchor_grid = load_anchors_and_grid(anchors, anchor_grid)

    model = Chain(                                                                                                                                                                                  
        YOLOv7.YOLOv7Backbone(p3=true, p4=true),
        YOLOv7.SPPCSPC(1024, 512),
        YOLOv7.YOLOv7HeadRouteback(512, :p4),
        YOLOv7.YOLOv7HeadBlock(256, :h1), # 63
        YOLOv7.YOLOv7HeadRouteback(256, :p3),
        YOLOv7.YOLOv7HeadBlock(128, :h2), # 75
        YOLOv7.YOLOv7HeadIncep(128, :h1),
        YOLOv7.YOLOv7HeadBlock(256, :h3), #88
        YOLOv7.YOLOv7HeadIncep(256, :sppcspc),
        YOLOv7.YOLOv7HeadBlock(512, :h4), #101
        YOLOv7.YOLOv7HeadTailRepConv(128, :h2, :h3, :h4),
        YOLOv7.IDetec(nc; channels=(256, 512, 1024), anchors=anchors, anchor_grid=anchor_grid),
    )

    return yolov7(name, false, class_names, model)
end

function yolov7_from_torch(;name="yolov7", pickle_path="pretrain/yolov7_training.pt", nc=80, class_names=[], anchors=(), anchor_grid=(), channels=())
    d::OrderedDict, class_names_model, nc_model = load_pickle(pickle_path)

    keep_head = false
    if nc_model == nc
        keep_head = true
        class_names = class_names_model
    else
        @assert nc == length(class_names) "Number of classes (nc: $nc) need to match the number of class names ($(length(class_names)))"
    end

    anchors, anchor_grid = load_anchors_and_grid(anchors, anchor_grid)

    model = Chain(                                                                                                                                                                                  
        YOLOv7.YOLOv7Backbone(d; p3=true, p4=true),
        YOLOv7.SPPCSPC(d),
        YOLOv7.YOLOv7HeadRouteback(512, :p4, d; off=0),
        YOLOv7.YOLOv7HeadBlock(256, :h1, d; off=0), # 63
        YOLOv7.YOLOv7HeadRouteback(256, :p3, d; off=12),
        YOLOv7.YOLOv7HeadBlock(128, :h2, d; off=12), # 75
        YOLOv7.YOLOv7HeadIncep(128, :h1, d; off=0),
        YOLOv7.YOLOv7HeadBlock(256, :h3, d; off=25), #88
        YOLOv7.YOLOv7HeadIncep(256, :sppcspc, d; off=13),
        YOLOv7.YOLOv7HeadBlock(512, :h4, d; off=38), #101
        keep_head ? YOLOv7.YOLOv7HeadTailRepConv(128, :h2, :h3, :h4, d) : YOLOv7.YOLOv7HeadTailRepConv(128, :h2, :h3, :h4),
        keep_head ? YOLOv7.IDetec(d) : YOLOv7.IDetec(nc; anchors=anchors, anchor_grid=anchor_grid, channels=channels),
    )

    return yolov7(name, true, class_names, model)
end

function load_anchors_and_grid(anchors, anchor_grid)
    if isempty(anchor_grid) && isempty(anchors)
        anchor_grid = [
            [[12 16]; [19 36]; [40 28]],
            [[36 75]; [76 55]; [72 146]],
            [[142 110]; [192 243]; [459 401]]
        ]

        anchors = anchor_grid ./ [8, 16, 32]
    elseif isempty(anchor_grid)
        @assert length(anchors) == 3

        anchor_grid = anchors .* [8, 16, 32]
    elseif isempty(anchors)
        @assert length(anchor_grid) == 3

        anchors = anchor_grid ./ [8, 16, 32]
    else
        @assert length(anchor_grid) == 3
        @assert length(anchors) == 3
    end

    return anchors, anchor_grid
end
