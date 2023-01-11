using NNlib: logσ
using Flux: epseltype
using CUDA
using Statistics: mean
using ChainRulesCore

function xlogyσ(x::Number, y::Number)
    result = x * logσ(y)
    return iszero(x) && !isnan(y) ? zero(result) : result
end

function xlogy(x::Number, y::Number)
    result = x * log(y)
    return iszero(x) && !isnan(y) ? zero(result) : result
end

function _check_sizes(ŷ::AbstractArray, y::AbstractArray)
    for d in 1:max(ndims(ŷ), ndims(y)) 
     size(ŷ,d) == size(y,d) || throw(DimensionMismatch(
        "loss function expects size(ŷ) = $(size(ŷ)) to match size(y) = $(size(y))"
      ))
    end
  end
  _check_sizes(ŷ, y) = nothing  # pass-through, for constant label e.g. y = 1
  
  ChainRulesCore.@non_differentiable _check_sizes(ŷ::Any, y::Any)

function logitbinarycrossentropy(ŷ, y; agg = mean, ϵ = epseltype(ŷ), pos_weights=nothing)
_check_sizes(ŷ, y)
    if pos_weights === nothing
        agg(@.(-xlogyσ(y, ŷ + ϵ) - xlogyσ(1 - y, 1 - ŷ + ϵ)))
    else
        agg(@.((-pos_weights*xlogyσ(y, ŷ + ϵ) - xlogyσ(1 - y, 1 - ŷ + ϵ))))
    end
end

struct ComputeLoss
    hyper::Dict{String,Any}
    BCEclass::Function
    BCEobj::Function
    autobalance::Bool
    na::Int
    nl::Int
    anchors::Tuple
    gr::Float32
end

function ComputeLoss(hyper::Dict{String,Any}, model::Flux.Chain; autobalance::Bool=false, gr=1.0f0)
    BCEclass = (ŷ, y) -> logitbinarycrossentropy(ŷ, y; agg = mean, ϵ = epseltype(ŷ), pos_weights=hyper["cls_pw"])
    BCEobj = (ŷ, y) -> logitbinarycrossentropy(ŷ, y; agg = mean, ϵ = epseltype(ŷ), pos_weights=hyper["obj_pw"])
    
    det = model[end]
    balance = get(Dict(3 => [4.0, 1.0, 0.4]), det.detec_layers, [4.0, 1.0, 0.25, 0.06, 0.02])
    # TODO: ssi correction
    ssi = autobalance ? 1 : 0

    na = det.n_anchors
    # nc
    nl = det.detec_layers
    anchors = det.anchors

    return ComputeLoss(hyper, BCEclass, BCEobj, autobalance, na, nl, anchors, gr)
end

function build_targets(compute_loss::ComputeLoss, ŷ, y; device=gpu)
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)

    na, nt = compute_loss.na, size(y)[2]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    
    gain = Flux.ones32(7) |> device # normalized to gridspace gain
    ai = repeat(range(1, na), 1, nt) |> device
    ai = permutedims(ai, (2, 1))

    targets = cat(repeat(y, 1, 1, na), reshape(ai, (1, size(ai)...)), dims=1) |> device # append anchor indices

    println("targets ", size(targets))

    # final targets = (7, 4, 3)
    g = 0.5  # bias
    off = [[0  0];
           [1  0]; [0  1]; [-1  0]; [0  -1]  # j,k,l,m
         # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
        ]' .* g  # offsets
    
    for i in 1:compute_loss.nl
        anchors = compute_loss.anchors[i] |> device
        println("size(anchors) ", size(anchors))
        println("anchors ", anchors)
        println("size(anchors') ", size(anchors'))
        println("anchors' ", anchors')

        println(size(ŷ[i]))
        println(size(ŷ[i])[[2, 3, 2, 3]])
        gain[3:6] .= size(ŷ[i])[[2, 3, 2, 3]] |> device  # xyxy gain

        # Match targets to anchors
        t = targets .* gain
        println("size(t) ", size(t))
        println("t1", t[1, :])

        if nt != 0
            # Matches
            r = t[5:6, :, :] ./ reshape(anchors', 2, 1, 3)  # wh ratio
            println("rsize ", size(r))
            
            j = max.(r, 1.f0 ./ r)
 
            println("jsize ", size(j))
            println(j)

            
            
            j = maximum(j, dims=1)[1, :, :] .< compute_loss.hyper["anchor_t"]  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            println("jsize after filter", size(j))
            t = t[:, j]  # filter
            println("t1", t[1, :])

            println("tsize after filter", size(t))
            println("t ", t)
            # Offsets
            gxy = t[3:4, :]  # grid xy
            println("gxy ", size(gxy))
            println("gain[[3, 4]] ", size(gain[[3, 4]]))
            gxi = gain[[3, 4]] .- gxy  # inverse
            println("gxi ", size(gxi))

            println("gxy: ", gxy)
            println("typeof(gxy) ", typeof(gxy))

            # j, k = @. ((gxy % 1.0f0 < g) & (gxy > 1.0f0))
            jk = @. ((gxy % 1.0f0 < g) & (gxy > 1.0f0))
            j = jk[1, :]
            k = jk[2, :]

            println("j ", size(j))
            println("k ", size(k))

            lm = @. ((gxi % 1.0f0 < g) & (gxi > 1.0f0))
            l = lm[1, :]
            m = lm[2, :]

            println("l ", size(l))
            println("m ", size(m))


            j = Bool.(cat(Flux.ones32(size(j)...) |> device, j, k, l, m; dims=2))
            println("size(j) ", size(j))
            println("j ", j)


            t = repeat(t, 1, 1, 5)
            println("size(t) ", size(t))
            println("typeof(t) ", typeof(t))
            t = t[:, j]
            println("t1", t[1, :])
            println("size(t) ", size(t))
            println("typeof(t) ", typeof(t))

            println("size(off) ", size(off))
            println("size(gxy) ", size(gxy))

            println(size(Flux.zeros32(size(gxy)..., size(off)[2])))
            println(size(repeat(off, 1, 1, size(gxy)[2])))
            perm_off = permutedims(repeat(off, 1, 1, size(gxy)[2]), (1, 3, 2)) |> device
            println(size(perm_off))

            offsets = (zeros(size(gxy)..., size(off)[2]) |> device) .+ perm_off
            println("size(offsets) ", size(offsets))
            offsets = offsets[:, j]
            println("size(offsets) ", size(offsets))
        else
            t = targets[0]
            offsets = 0
        end
        
        # Define
        println("size(t) ", size(t))
        b = Int.(trunc.(t[1, :]))
        c = Int.(trunc.(t[2, :]))  # image, class
        gxy = t[3:4, :]  # grid xy
        gwh = t[5:6, :]  # grid wh
        gij = Int.(trunc.(gxy - offsets))
        gi = gij[1, :]  # grid xy indices
        gj = gij[2, :]

        # Append
        a = Int.(trunc.(t[7, :]))  # anchor indices
        println(a)
        # indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        CUDA.@allowscalar push!(indices, [CartesianIndex(i, j, k, l) for (i, j, k, l) in zip(
            clamp!(gi, 1, gain[3] |> cpu),
            clamp!(gj, 1, gain[4] |> cpu),
            a,
            b)])

        # tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        println("size(gwh) ", size(gwh))
        println("size(gxy - gij) ", size(gxy - gij))
        push!(tbox, cat(gxy - gij, gwh, dims=1))


        # anch.append(anchors[a])  # anchors
        println("size(anchors) ", size(anchors))
        println("size(a) ", size(a))
        CUDA.@allowscalar push!(anch, CUDA.@allowscalar anchors[[Int.(a)]...])

        # tcls.append(c)  # class
        push!(tcls, c)
    end

    return tcls, tbox, indices, anch
end

function bbox_iou(box1, box2; x1y1x2y2=true, GIoU=false, DIoU=false, CIoU=false, eps=1e-7)
    println("size(box1)", size(box1))
    println("size(box2)", size(box2))

    if x1y1x2y2
        return 0
    else
        b1_x1, b1_x2 = box1[1, :] - box1[3, :] / 2.0f0, box1[1, :] + box1[3, :] / 2.0f0
        b1_y1, b1_y2 = box1[2, :] - box1[4, :] / 2.0f0, box1[2, :] + box1[4, :] / 2.0f0
        b2_x1, b2_x2 = box2[1, :] - box2[3, :] / 2.0f0, box2[1, :] + box2[3, :] / 2.0f0
        b2_y1, b2_y2 = box2[2, :] - box2[4, :] / 2.0f0, box2[2, :] + box2[4, :] / 2.0f0
    end

    println("size(b1_x1) ", size(b1_x1))
    println("size(b1_y1) ", size(b1_y1))
    println("size(b2_x2) ", size(b2_x2))
    println("size(b2_y2) ", size(b2_y2))

    inter = clamp.(min.(b1_x2, b2_x2) .- max.(b1_x1, b2_x1), 0, Inf) .* clamp.(min.(b1_y2, b2_y2) .- max.(b1_y1, b2_y1), 0, Inf)
    println("size(inter)", size(inter))

    w1, h1 = @. b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = @. b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = @. w1 * h1 + w2 * h2 - inter + eps
    println("size(union)", size(union))

    iou = inter ./ union

    if GIoU || DIoU || CIoU
        cw = @. max(b1_x2, b2_x2) - min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = @. max(b1_y2, b2_y2) - min(b1_y1, b2_y1)  # convex height
        println(size(cw), " ", size(ch))

        if CIoU || DIoU
            c2 = @. cw ^ 2 + ch ^ 2 + eps  # convex diagonal squared
            rho2 = @. ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ^ 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ^ 2) / 4
            println(size(c2), " ", size(rho2))
            if DIoU
                return @. iou - rho2 / c2  # DIoU
            elseif CIoU  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = @. (4 / pi ^ 2) * (atan(w2 / (h2 + eps)) - atan(w1 / (h1 + eps))) ^ 2
                println("size(v) ", size(v))
                alpha = 1
                ChainRulesCore.@ignore_derivatives alpha = @. v / (v - iou + (1 + eps))
                return @. iou - (rho2 / c2 + v * alpha)
            end
        else
        end

    else
    end
end

function loss(compute_loss::ComputeLoss, ŷ, y, bs; device=gpu, nc=5, cn=0.0f0, cp=1.0f0, balance=[4.0, 1.0, 0.4])
    lcls, lbox, lobj = 0f0, 0f0, 0f0
    tcls, tbox, indices, anchors = build_targets(compute_loss, ŷ, y)  # targets

    for (i, p_i) in enumerate(ŷ)
        tobj = Flux.zeros32(size(p_i)[2:end-1]..., 1) |> device # target obj

        println("size(tobj) ", size(tobj))

        n = size(indices[i])[1]
        println("n ", n)

        if n > 0
            println("size(p_i) ", size(p_i))

            ps = p_i[:, indices[i]] 
            println("size(ps) ", size(ps)) # prediction subset corresponding to targets
            println("ps ", ps)

            pxy = sigmoid(ps[1:2, :]) .* 2. .- 0.5
            println("size(pxy)", size(pxy))

            pwh = (sigmoid(ps[3:4, :]) .* 2) .^ 2 .* anchors[i]'
            println("size(pwh)", size(pwh))

            pbox = cat(pxy, pwh; dims=1)
            println("size(pbox)", size(pbox))

            iou = bbox_iou(pbox, tbox[i]; x1y1x2y2=false, CIoU=true)

            println("size(iou) ", size(iou))

            lbox += mean(1.0 .- iou)

            # Objectness
            println(size(tobj))
            println("size(tobj[indices[i]]) ", size(tobj[indices[i]]))
            println(size((1.0 - compute_loss.gr) .+ compute_loss.gr .* clamp.(iou, 0.0, Inf)))
            tobj[indices[i]] .= (1.0 - compute_loss.gr) .+ compute_loss.gr .* clamp.(iou, 0.0, Inf)

            # Classification
            if nc > 1
                t = fill(cn, size(ps[6:end, :])) |> device
                println("size(t) ", size(t))
                t[tcls[i], 1:n] .= cp

                lcls += compute_loss.BCEclass(ps[6:end, :], t)
            end
        end

        println("size(tobj) ", size(tobj))
        println("size(p_i[5, :, :, :, :]) ", size(p_i[5, :, :, :, :]))

        obji = compute_loss.BCEobj(p_i[5, :, :, :, :], tobj)
        lobj += obji * balance[i] 
    end

    lbox *= compute_loss.hyper["box"]
    lobj *= compute_loss.hyper["obj"]
    lcls *= compute_loss.hyper["cls"]

    loss = lbox + lobj + lcls
    return loss * bs, [lbox, lobj, lcls, loss]
end
