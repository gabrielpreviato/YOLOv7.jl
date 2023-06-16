using Zygote

# function withgradient(f, args...)
#     y_a, back = pullback(f, args...)
#     y = y_a[1]
#     s = Zygote.sensitivity(y)
#     grad = back((s, 0, 0, 0, 0))
#     results = isnothing(grad) ? map(_ -> nothing, args) : map(Zygote._project, args, grad)
#     (val=y_a, grad=results)
#   end