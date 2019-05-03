include("../ccsa_quadratic_sparse.jl")
using CSV

function myfunc(x::Vector{Float64}, grad::Vector{Float64}, A::Matrix{Float64}, y::Vector{Float64})::Float64
    # minimizing (x[1] - 2)^2 + abs(x)
    # replacing abs(x) with t subject to x[1]-t<=0 and -x[1]-t <= 0
    @assert length(x) % 2 == 0
    n = length(x)
    n_div = convert(Int, n/2)
    error = y .- A*x[1:n_div]
    grad_copy = -2.0 * transpose(A) * error
    if length(grad) > 0
        for i =1:n
            if i <= n/2
                grad[i] = copy(grad_copy[i])
            else
                grad[i] = 1.0
            end
        end
    end
    return sum( error.^2 ) + sum(x[(n_div+1):n])
end

function myconstraints(x::Vector{Float64}, dfcdx::SparseMatrixCSC{Float64, Int64})::Vector{Float64}
    # minimizing (x[1] - 2)^2 + abs(x)
    # replacing abs(x) with t subject to x[1]-t<=0 and -x[1]-t <= 0
    n = length(x)
    @assert n % 2 == 0
    n_div = convert(Int, n/2)
    result = zeros(Float64, n)
    if length(dfcdx) > 0
        if count(!iszero, dfcdx) != 2*n
            # need to change it, otherwise I am good
            for i=1:n_div
                dfcdx[2*i - 1,i] = 1.0
                dfcdx[2*i - 1, n_div+i] = -1.0
                dfcdx[2*i, i] = -1.0
                dfcdx[2*i, n_div+i] = -1.0
                result[2*i - 1] = x[i] - x[n_div+i]
                result[2*i] = -x[i] - x[n_div+i]
            end
            @assert count(!iszero, dfcdx) == 2*n
            return result
        end
    end

    for i=1:n_div
        result[2*i - 1] = x[i] - x[n_div+i]
        result[2*i] = -x[i] - x[n_div+i]
    end

    return result
end

data = CSV.read("/Users/davidhunter/nlopt/code_test_nlopt_project/data_as_csv/boston.csv", header=false)
samples = size(data)[1]
n = size(data)[2]
print(size(data))
A = convert(Matrix{Float64}, data[:,1:(n-1)])
y = convert(Vector{Float64}, data[:,n])
n -= 1
n *= 2
n_div = convert(Int, n/2)

lb = fill(-Inf, n)
ub = fill(Inf,n)
x = zeros(Float64, n)
x = ccsa_quadratic_minimize(n, (x,grad) -> myfunc(x,grad,A,y), n, myconstraints, lb, ub, x)
println(x[1:n_div])
println("\n")
println(x[(n_div+1):n])
println("\n")
println(myconstraints(x, spzeros(n,n)))
println(myfunc(x, zeros(0),A,y))
