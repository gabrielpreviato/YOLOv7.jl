using NNlib: logσ
using Flux: epseltype
using CUDA
using Statistics: mean
using ChainRulesCore
using LinearAlgebra

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
    anchors
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
    # tcls, tbox, indices, anch = [], [], [], []
    
    gain = Flux.ones32(7) |> device # normalized to gridspace gain
    ai = repeat(range(1, na), 1, nt) |> device
    ai = permutedims(ai, (2, 1))

    targets = cat(repeat(y, 1, 1, na), reshape(ai, (1, size(ai)...)), dims=1) |> device # append anchor indices

    # println(targets[:, :, 1])

    # final targets = (7, 4, 3)
    g = 0.5  # bias
    off = [[0  0];
           [1  0]; [0  1]; [-1  0]; [0  -1]  # j,k,l,m
         # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
        ]' .* g  # offsets
    
    # Zygote.@showgrad(off)
    
    ret = Zygote.@ignore [
        let
            anchors = compute_loss.anchors[i] |> device

            # Zygote.@showgrad(anchors)
            # println("size(anchors) ", size(anchors))
            # println("anchors ", anchors)
            # println("size(anchors') ", size(anchors'))
            # println("anchors' ", anchors')

            # println(size(ŷ[i]))
            # println(size(ŷ[i])[[2, 3, 2, 3]])

            # println("before gain ", size(gain), typeof(gain), gain)
            
            # new_gain = Zygote.Buffer(gain)
            # new_gain = gain
            # new_gain[3:6] = size(ŷ[i])[[2, 3, 2, 3]]  # xyxy gain
            # gain = copy(new_gain)

            gain = CUDA.@allowscalar [gain[1:2]..., size(ŷ[i])[[2, 3, 2, 3]]..., gain[7]] |> device

            # println("gain ", size(gain), typeof(gain), gain)

            # Match targets to anchors
            # println("size(targets) ", size(targets))
            # println("size(gain) ", size(gain))
            t = targets .* gain

            # Zygote.@showgrad(t)
            # println("size(t) ", size(t))
            # println("t1", t[:, :, 1])

            if nt != 0
                # Matches
                r = t[5:6, :, :] ./ reshape(anchors', 2, 1, 3)  # wh ratio
                # println("rsize ", size(r))
                
                j = max.(r, 1.f0 ./ r)
    
                # println("jsize before filter ", size(j))
                # println(j)

                
                j = maximum(j, dims=1)[1, :, :]

                # print("j to be compared ", j)

                j = j .< compute_loss.hyper["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # println("jsize after filter ", size(j))
                
                # println("tsize before filter ", size(t))
                t = t[:, j]  # filter
                # println("t1 after filter", t[:, :, 1])

                # println("tsize after filter ", size(t))
                # println("t ", t)
                if size(t)[end] != 0
                    # Offsets
                    gxy = t[3:4, :]  # grid xy
                    # println("gxy ", gxy)
                    # println("gain[[3, 4]] ", size(gain[[3, 4]]))
                    gxi = gain[[3, 4]] .- gxy  # inverse
                    # println("gxi ", size(gxi))

                    # println("gxi: ", gxi)
                    # println("typeof(gxy) ", typeof(gxy))

                    # j, k = @. ((gxy % 1.0f0 < g) & (gxy > 1.0f0))
                    jk = @. ((gxy % 1.0f0 < g) & (gxy > 1.0f0))
                    j = jk[1, :]
                    k = jk[2, :]

                    # println("jk ", jk)
                    # println("k ", size(k))

                    lm = @. ((gxi % 1.0f0 < g) & (gxi > 1.0f0))
                    l = lm[1, :]
                    m = lm[2, :]

                    # println("lm ", lm)

                    # println("l ", size(l))
                    # println("m ", size(m))


                    j = Bool.(cat(Flux.ones32(size(j)...) |> device, j, k, l, m; dims=2))
                    # println("size(j) ", size(j))
                    # println("j ", j)


                    t = repeat(t, 1, 1, 5)
                    # println("size(t) ", size(t))
                    # println("typeof(t) ", typeof(t))
                    t = t[:, j]
                    # println("t: ", t[:, :])
                    # println("size(t) ", size(t))
                    # println("typeof(t) ", typeof(t))

                    # println("size(off) ", size(off))
                    # println("size(gxy) ", size(gxy))

                    # println(size(Flux.zeros32(size(gxy)..., size(off)[2])))
                    # println(size(repeat(off, 1, 1, size(gxy)[2])))
                    perm_off = permutedims(repeat(off, 1, 1, size(gxy)[2]), (1, 3, 2)) |> device
                    # println(size(perm_off))

                    offsets = (zeros(size(gxy)..., size(off)[2]) |> device) .+ perm_off
                    # println("size(offsets) ", size(offsets))
                    offsets = offsets[:, j]
                    # println("size(offsets) ", size(offsets))
                else
                    t = targets[:, 1, :]
                    # println(size(targets))
                    offsets = 0
                end
            else
                t = targets[:, 1, :]
                # println(size(targets))
                offsets = 0
            end
            
            # Define
            # println("size(t) ", size(t))
            b = Int.(trunc.(t[1, :]))
            c = Int.(trunc.(t[2, :]))  # image, class
            gxy = t[3:4, :]  # grid xy
            gwh = t[5:6, :]  # grid wh

            # println(size(t))
            

            gij = Int.(trunc.(gxy .- offsets))
            gi = gij[1, :]  # grid xy indices
            gj = gij[2, :]

            # Append
            a = Int.(trunc.(t[7, :]))  # anchor indices
            # println(a)
            
            ix = CUDA.@allowscalar Int.(min.(max.(gi, 1), gain[3] |> cpu))
            ij = CUDA.@allowscalar Int.(min.(max.(gj, 1), gain[4] |> cpu))

            indices = CUDA.@allowscalar [CartesianIndex(i, j, k, l) for (i, j, k, l) in zip(
                ix,
                ij,
                a,
                b)]

            # tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            # println("size(gwh) ", size(gwh))
            # println("size(gxy - gij) ", size(gxy - gij))
            # push!(tbox, cat(gxy - gij, gwh, dims=1))


            # anch.append(anchors[a])  # anchors
            # println("size(anchors) ", size(anchors))
            # println("size(a) ", size(a))
            anch = CUDA.@allowscalar anchors[[Int.(a)]...]

            # tcls.append(c)  # class
            # push!(tcls, c)
            # tcls, tbox, indices, anch
            pp = cat(gxy - gij, gwh, dims=1)

            # Zygote.@showgrad(pp)

            [c, pp, indices, anch]
        end
        for i in 1:compute_loss.nl
    ]

    tcls = [i[1] for i in ret]
    # println(size.(tcls), typeof(tcls))
    # println(size(tcls[1]), typeof(tcls[1]))

    tbox = [i[2] for i in ret]
    indices = [i[3] for i in ret]
    anch = [i[4] for i in ret]

    return tcls, tbox, indices, anch
end

function bbox_iou(box1, box2; x1y1x2y2=true, GIoU=false, DIoU=false, CIoU=false, eps=Float32(1e-5))
    # println("size(box1)", size(box1), typeof(box1))
    # println("size(box2)", size(box2), typeof(box2))

    if x1y1x2y2
        b1_x1, b1_x2 = box1[1, :], box1[3, :]
        b1_y1, b1_y2 = box1[2, :], box1[4, :]
        b2_x1, b2_x2 = box2[1, :], box2[3, :]
        b2_y1, b2_y2 = box2[2, :], box2[4, :]
    else
        b1_x1, b1_x2 = @. box1[1, :] - box1[3, :] / 2.0f0, box1[1, :] + box1[3, :] / 2.0f0
        b1_y1, b1_y2 = @. box1[2, :] - box1[4, :] / 2.0f0, box1[2, :] + box1[4, :] / 2.0f0
        b2_x1, b2_x2 = @. box2[1, :] - box2[3, :] / 2.0f0, box2[1, :] + box2[3, :] / 2.0f0
        b2_y1, b2_y2 = @. box2[2, :] - box2[4, :] / 2.0f0, box2[2, :] + box2[4, :] / 2.0f0
    end

    # println("size(b1_x1) ", size(b1_x1))
    # println("size(b1_y1) ", size(b1_y1))
    # println("size(b2_x2) ", size(b2_x2))
    # println("size(b2_y2) ", size(b2_y2))

    inter = max.(min.(b1_x2, b2_x2) .- max.(b1_x1, b2_x1), 0) .* max.(min.(b1_y2, b2_y2) .- max.(b1_y1, b2_y1), 0)
    
    # Zygote.@showgrad(inter)
    # println("size(inter)", size(inter))

    w1, h1 = @. b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = @. b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = @. w1 * h1 + w2 * h2 - inter + eps
    # println("size(union)", size(union))

    iou = inter ./ union

    if GIoU || DIoU || CIoU
        cw = @. max(b1_x2, b2_x2) - min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = @. max(b1_y2, b2_y2) - min(b1_y1, b2_y1)  # convex height
        # println(size(cw), " ", size(ch))

        if CIoU || DIoU
            c2 = @. cw ^ 2 + ch ^ 2 + eps  # convex diagonal squared
            rho2 = @. ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ^ 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ^ 2) / 4
            # println(size(c2), " ", size(rho2))
            if DIoU
                return @. iou - rho2 / c2  # DIoU
            elseif CIoU  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = Float32(4.0f0 / pi ^ 2) .* (atan.(w2 ./ (h2 .+ eps)) .- atan.(w1 ./ (h1 .+ eps))) .^ 2
                # println("size(v) ", size(v))
                # alpha = 1
                # ChainRulesCore.@ignore_derivatives alpha = @. v / (v - iou + (1 + eps))
                alpha = @. v / (v - iou + (1 + eps))
                return @. iou - (rho2 / c2 + v * alpha)
            end
        else
            return 0
        end

    else
        return iou
    end
end

function loss(compute_loss::ComputeLoss, ŷ, y, bs; device=gpu, nc=5, cn=0.0f0, cp=1.0f0, balance=[4.0, 1.0, 0.4])
    lcls, lbox, lobj = 0f0, 0f0, 0f0
    tcls, tbox, indices, anchors = Zygote.@ignore build_targets(compute_loss, ŷ, y)  # targets

    # Zygote.@showgrad(tbox)

    for (i, p_i) in enumerate(ŷ)
        tobj = Flux.zeros32(size(p_i)[2:end-1]..., bs) |> device # target obj

        # println("size(tobj) ", size(tobj))

        n = size(indices[i])[1]
        counter_indices = filter(x -> !(x in indices[i]), CartesianIndices(tobj))
        # index = Zygote.Buffer(indices[i])
        # index[indices[i]] = 1
        # println("n ", n)

        if n > 0
            # println("size(p_i) ", size(p_i), typeof(p_i))

            ps = p_i[:, indices[i]] 
            # println("size(ps) ", size(ps), typeof(ps)) # prediction subset corresponding to targets

            pxy = sigmoid(ps[1:2, :]) .* 2.0f0 .- 0.50f0
            # println("size(pxy)", size(pxy), typeof(pxy))

            # println("size(reshape(anchors[i], size(anchors[i]), 1))", size(reshape(anchors[i], size(anchors[i])[1], 1)), typeof(reshape(anchors[i], size(anchors[i])[1], 1)))
            pwh = (sigmoid(ps[3:4, :]) .* 2.0f0) .^ 2 .* reshape(anchors[i], 1, size(anchors[i])[1])
            # println("size(pwh)", size(pwh), typeof(pwh))

            pbox = cat(pxy, pwh; dims=1)
            # println("size(pbox)", size(pbox))

            iou = bbox_iou(pbox, tbox[i]; x1y1x2y2=false, CIoU=true)

            # println("size(iou) ", size(iou))

            lbox += mean(1.0f0 .- iou)

            # Objectness
            # println(size(tobj))
            # println("size(tobj[indices[i]]) ", size(tobj[indices[i]]))
            obji_no_change = compute_loss.BCEobj(p_i[5, indices[i]], (1.0f0 - compute_loss.gr) .+ compute_loss.gr .* max.(iou, 0.0f0))

            obji_changed = compute_loss.BCEobj(p_i[5, counter_indices], tobj[counter_indices])
            # Classification
            if nc > 1
                # t = fill(cn, size(ps[6:end, :])) |> device

                id = Matrix(cp*I, nc, nc) |> device
                # println("size(t) ", size(t))
                # println(tcls[i])
                # t[1:n, tcls[i]] .= cp

                t = reduce(hcat, [
                    CUDA.@allowscalar id[:, tcls[i][j]]
                    for j in 1:n
                ])


                lcls += compute_loss.BCEclass(ps[6:end, :], t)
            end
        end

        # println("size(tobj) ", size(tobj))
        # println("size(p_i[5, :, :, :, :]) ", size(p_i[5, :, :, :, :]))

        obji = (obji_no_change*length(indices[i]) + obji_changed*length(counter_indices)) / length(tobj)
        lobj += obji * balance[i] 
    end

    lbox *= compute_loss.hyper["box"]
    lobj *= compute_loss.hyper["obj"]
    lcls *= compute_loss.hyper["cls"]

    loss = lbox + lobj + lcls
    return loss * bs , Zygote.@ignore [lbox, lobj, lcls, loss]
end
