function output_to_box(ŷ, anchors_grid, stride)
    z = []
    

    for (i, ŷ_i) in enumerate(ŷ)
        xs, ys = size(ŷ_i)[2:3]
        bs = size(ŷ_i)[end]
        no = size(ŷ_i)[1]
        
        grid = reshape(
            stack([((0:xs-1)' .* ones(ys))', ((0:ys-1)' .* ones(xs))]; dims=1),
            (2, ys, xs, 1, 1))

        sig = sigmoid(ŷ_i) |> cpu

        sig[1:2, :, :, :, :] .= @. (sig[1:2, :, :, :, :] * 2.0 - 0.5 + grid) * stride[i]

        anch_g = reshape(anchors_grid[i]', (2, 1, 1, 3, 1))
        sig[3:4, :, :, :, :] .= @. (sig[3:4, :, :, :, :] * 2.0) ^ 2 
        sig[3:4, :, :, :, :] .= sig[3:4, :, :, :, :] .* anch_g

        push!(z, reshape(sig, (size(sig)[1], :, size(sig)[end])))
    end
    
    hcat(z...)
end

function xywh2xyxy(x)
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = copy(x)
    y[1, :] .= @. x[1, :] - x[3, :] / 2  # top left x
    y[2, :] .= @. x[2, :] - x[4, :] / 2  # top left y
    y[3, :] .= @. x[1, :] + x[3, :] / 2  # bottom right x
    y[4, :] .= @. x[2, :] + x[4, :] / 2  # bottom right y
    return y
end

function nms(boxes, scores, iou_thres)
    detec::Array{Int} = []
    boxes_cp = []
    scores_cp = copy(scores)
    for i in 1:size(boxes)[2]
        push!(boxes_cp, boxes[:, i])
    end

    while sum(scores_cp) > 0
        i = argmax(scores_cp)

        add = true
        for d in detec
            iou = bbox_iou(boxes[:, d], boxes[:, i])[1]
            if iou > iou_thres
                add = false
                break
            end
        end

        if add
            push!(detec, i)
        end

        scores_cp[i] = 0
    end

    return detec
end

function non_max_suppression(prediction; nc=1, conf_thres=0.25, iou_thres=0.45, classes=nothing, agnostic=false, multi_label=false, labels=())
    """Runs Non-Maximum Suppression (NMS) on inference results
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = true  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = false  # use merge-NMS

    output = repeat([Flux.zeros32(0, 6)], size(prediction)[1])

    for (i, x) in enumerate(prediction)  # image index, image inference
        # Apply constraints
        nc = size(x)[1] - 5  # number of classes
        xc = x[5, :, :] .> conf_thres  # candidates
        x = x[:, :, :]

        x = permutedims(x, ((2:ndims(x) |> collect)..., 1))

        x = x[xc, :]'  # confidence

        # If none remain process next image
        if size(x)[1] == 0
            continue
        end

        # Compute conf
        if nc == 1
            x[6:end, :] = x[5, :] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else
            x[6:end, :] .= x[6:end, :] .* x[5, :]'  # conf = obj_conf * cls_conf
        end

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[1:4, :])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label
            # TODO: Convert this PyTorch
            # i, j = (x[:, 5:end] > conf_thres).nonzero(as_tuple=False).T
            # x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else  # best class only

            conf, j = findmax(x[6:end, :], dims=1)

            x = vcat(box, conf, map(x -> float.(x[1]), j))'
            x = x[conf[1, :] .> conf_thres, :]'
        end

        # Filter by class
        # TODO: Convert this PyTorch
        # if classes !== nothing
        #     x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        # end

        # Apply finite constraint

        # Check shape
        n = size(x)[2]  # number of boxes
        if n == 0  # no boxes
            continue
        elseif n > max_nms  # excess boxes
            x = x[sortperm(x[5, :])[1:max_nms]]  # sort by confidence
        end

        # Batched NMS
        c = x[6, :] .* (agnostic ? 0 : max_wh)  # classes]

        boxes = x[1:4, :] .+ c'   # boxes (offset by class) 
        scores = x[5, :]        # scores
        i_nms = nms(boxes, scores, iou_thres)  # NMS
        if length(i_nms) > max_det  # limit detections
            i_nms = i_nms[1:max_det]
        end
        
        # TODO: Convert this PyTorch
        # if merge && (1 < n < 3E3)  # Merge NMS (boxes merged using weighted mean)
        #     # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #     iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
        #     weights = iou * scores[None]  # box weights
        #     x[i_nms, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #     if redundant
        #         i_nms = i_nms[iou.sum(1) > 1]
        #     end
        # end  # require redundancy

        output[i] = x[:, i_nms]
    end

    return output
end