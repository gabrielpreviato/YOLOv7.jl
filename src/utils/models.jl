using YOLOv7

struct YOLOv7
    input
    block1
    block3
    block12
    block16
    block25
    block38
end

function YOLOv7()
    input = YOLOv7.Conv(3 => 32, 3, 1)

    block1 = Flux.Chain(
        YOLOv7.Conv(32 => 64, 3, 2),
        YOLOv7.Conv(64 => 64, 3, 1),
    )

    block3 = Flux.Chain(
        YOLOv7.Conv(64 => 128, 3, 2),
        Flux.Parallel(
            YOLOv7.Conv(128 => 64, 1, 1),
            Flux.Parallel(
                YOLOv7.Conv(128 => 64, 1, 1),
                YOLOv7.Conv(128 => 64, 3, 1),
            )
        )
    )
    
    
end

struct B3

end