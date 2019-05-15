using CSV
using JuMP
using Gurobi

# Get the data
data = CSV.read("/Users/davidhunter/nlopt/code_test_nlopt_project/data_as_csv/boston.csv", header=false)
samples = size(data)[1]
n = size(data)[2]
A = convert(Matrix{Float64}, data[:,1:(n-1)])
y = convert(Vector{Float64}, data[:,n])
n -= 1
n *= 2
n_div = convert(Int, n/2)

# Set up problem
model = Model(with_optimizer(Gurobi.Optimizer))
sense = MOI.MIN_SENSE
@variable(model, x[1:n], start=0.0)
@objective(model, sense, sum((y - A*x[1:n_div]).^2 ) + sum(10000.0*x[(n_div+1):n]) )
@constraint(model, con1[i = 1:n_div], x[i] <= x[n_div+i])
@constraint(model, con2[i = 1:n_div], -x[i] <= x[n_div+i])
JuMP.optimize!(model)
x = JuMP.value.(x)
println(x[1:n_div])
println(x[(n_div+1):n])
