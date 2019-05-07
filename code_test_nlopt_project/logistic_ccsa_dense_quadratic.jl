using CSV
using NLopt

function myfunc(x::Vector, grad::Vector, A::Matrix{Float64}, y::Vector{Float64})
    @assert length(x) % 2 == 0
    n = length(x)
    n_div = convert(Int, n/2)
    neg_log_likelihood = 0.0
    grad_copy = zeros(n_div)
    prediction = A*x[1:n_div]
    if length(grad) > 0
        for j=1:(size(A)[1])
            if isapprox(y[j], 1.0)
                neg_log_likelihood += log(1.0 + exp(-prediction[j]))
            elseif isapprox(y[j], 0.0)
                neg_log_likelihood += log(1.0 + exp(prediction[j]))
            else
                println("ERROR!")
            end
            grad_copy -= (y[j] - (1.0 / (1.0 + exp(-prediction[j])))) * A[j,:]
        end
        for i =1:n
            if i <= n_div
                grad[i] = copy(grad_copy[i])
            else
                grad[i] = 1.0
            end
        end
    else
        for j=1:(size(A)[1])
            if isapprox(y[j], 1.0)
                neg_log_likelihood += log(1.0 + exp(-prediction[j]))
            elseif isapprox(y[j], 0.0)
                neg_log_likelihood += log(1.0 + exp(prediction[j]))
            else
                println("ERROR!")
            end
        end
    end
    println(neg_log_likelihood + sum(x[(n_div+1):n]))
    return neg_log_likelihood + sum(x[(n_div+1):n])
end

function myconstraints(result::Vector, x::Vector, grad::Matrix)
    # replacing abs(x) with t subject to x[1]-t<=0 and -x[1]-t <= 0
    n = length(x)
    @assert n % 2 == 0
    n_div = convert(Int, n/2)
    if length(grad) > 0
        # need to change it, otherwise I am good
        grad[:,:] = zeros(n,n)
        for i=1:n_div
            grad[i, 2*i - 1] = 1.0
            grad[n_div+i, 2*i - 1] = -1.0
            grad[i, 2*i] = -1.0
            grad[n_div+i, 2*i] = -1.0
            result[2*i - 1] = x[i] - x[n_div+i]
            result[2*i] = -x[i] - x[n_div+i]
        end
        return
    end

    for i=1:n_div
        result[2*i - 1] = x[i] - x[n_div+i]
        result[2*i] = -x[i] - x[n_div+i]
    end

    return
end

data = CSV.read("/Users/davidhunter/nlopt/code_test_nlopt_project/data_as_csv/breast_cancer.csv", header=false)
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
z = zeros(n)
# For some reason, without this things break
#z[(n_div+1):n] =ones(n_div)


opt = Opt(:LD_CCSAQ, n)
opt.lower_bounds = lb
opt.upper_bounds = ub
opt.xtol_abs = 1e-5
opt.min_objective = (x,grad) -> myfunc(x,grad,A,y)
inequality_constraint!(opt, myconstraints, fill(0.0,n)) # Doing this also helps otherwise it gets stuck


(minf,minx,ret) = optimize(opt, z)
println(ret)
println(minx[1:n_div])
println("\n")
println(minx[(n_div+1):n])
println("\n")
println(minf)
println("\n")

result = zeros(n)
grad = zeros(n,n)
myconstraints(result, minx, grad)
