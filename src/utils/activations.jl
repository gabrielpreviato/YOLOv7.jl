using Flux

silu(x) = σ(x) .* x

norm_sigmoid(x) = 2*σ(x) - 1