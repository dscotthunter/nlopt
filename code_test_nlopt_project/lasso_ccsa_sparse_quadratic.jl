include("../ccsa_quadratic_sparse.jl")

function myfunc(x::Vector{Float64}, grad::Vector{Float64})::Float64
    # minimizing (x[1] - 2)^2 + abs(x)
    # replacing abs(x) with t subject to x[1]-t<=0 and -x[1]-t <= 0
    if length(grad) > 0
        grad[1] = 2.0 * (x[1] - 2.0)
        grad[2] = 1.0
    end
    return (x[1] - 2.0)^2 + x[2]
end

function myconstraints(x::Vector{Float64}, dfcdx::SparseMatrixCSC{Float64, Int64})::Vector{Float64}
    # minimizing (x[1] - 2)^2 + abs(x)
    # replacing abs(x) with t subject to x[1]-t<=0 and -x[1]-t <= 0
    if length(dfcdx) > 0
        dfcdx[1,1] = 1
        dfcdx[1,2] = -1
        dfcdx[2,1] = -1
        dfcdx[2,2] = -1
    end
    result = zeros(Float64, 2)
    result[1] = x[1] - x[2]
    result[2] = -x[1] - x[2]
    return result
end


lb = [-Inf, -Inf]
ub = [Inf, Inf]
x = [0.0, 50.0]
x = ccsa_quadratic_minimize(2, myfunc, 2, myconstraints, lb, ub, x)
println(x)
