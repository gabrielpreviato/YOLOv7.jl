using YOLOv7, Statistics, StatsBase, Images, ImageDraw, ImageView, MLUtils, Printf

model = yolov7_from_torch()
m = model.m


colors = sample(values(Colors.color_names) |> collect, size(model.class_names))
colors = [RGB{Float32}(c./255.0f0...) for c in colors]

for img_name in ["bus", "horses", "zidane", "image1", "image2", "image3"]

    x = float.(Images.load.(["$(@__DIR__)/$(img_name).jpg"]))
    x = reshape_image.(x)
    x = channelview.(x)
    x = permutedims.(x, ((3, 2, 1),))
    x = stack(x, dims=4)

    ŷ = m(x)

    out = output_to_box(ŷ, model.anchor_grid, model.strid)
    out_nms = non_max_suppression([out]; nc=80, conf_thres=0.5)[1]

    out_nms_trunc = round.(Int, out_nms) 
    # for i in 1:size(out_nms_trunc)[2]
    #     x1, y1 = out_nms_trunc[1, i], out_nms_trunc[2, i]
    #     x2, y2 = out_nms_trunc[3, i], out_nms_trunc[4, i]
    #     draw!(img_rgb, Polygon(RectanglePoints(x1, y1, x2, y2)), colors[2])
    # end

    img = copy(x[:, :, :, 1])
    img_CHW = permutedims(img, (3, 2, 1))
    img_rgb = colorview(RGB, img_CHW)


    for i in 1:size(out_nms_trunc)[2]
        x1, y1 = out_nms_trunc[1, i], out_nms_trunc[2, i]
        x2, y2 = out_nms_trunc[3, i], out_nms_trunc[4, i]

        draw!(img_rgb, Polygon(RectanglePoints(x1, y1, x2, y2)), colors[out_nms_trunc[6,i]])

        fontsize=14
        s = @sprintf("%s: %.2f", model.class_names[out_nms_trunc[6,i]], out_nms[5,i])
        if y1-fontsize >= 2
            YOLOv7.BasicTextRender.overlaytext!(img_rgb, s, fontsize, (y1-fontsize, x1+1))
        else
            YOLOv7.BasicTextRender.overlaytext!(img_rgb, s, fontsize, (y1+1, x1+1))
        end
    end

    # gui = imshow(img_rgb)

    save("$(@__DIR__)/test/images/$(img_name)_pred.jpg", img_rgb)
end
