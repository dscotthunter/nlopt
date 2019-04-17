#=
In this file I implement Svanberg's CCSA algorithm with simple linear
approximation + quadratic penalty function, where I also make use of
Julia's sparse array package for the Jacobian of the constraints. This
implementation is based on Steven Johnson's nlopt implementation of
ccsa_quadratic. I only implement the non-preconditioned case based on the
Svanberg paper! I would also like to thank Steven Johnson for recommending
that I take advantage of the Julia sparse array package.
=#

using LinearAlgebra, SparseArrays

# Some user defined variables for the code. RHOMIN is the minimum value of
# rho, which I took from the paper.
ccsa_verbose = 0
CCSA_RHOMIN = 1e-5

# Basing this struct off of Steven Johnson's implementation
struct dual_data
    count_dual::Int # number of total inner + outer approximations
    n::Int # dimensions of problem
    x::Vector{Float64} # the vector of interest
    lb::Vector{Float64} # lower bounds for x
    ub::Vector{Float64} # upper bounds for x
    sigma::Vector{Float64}
    dfdx::Vector{Float64}
    dfcdx::SparseMatrixCSC{Float64, Int64} # This is m times n sparse Jacobian matrix
    fval::Float64 # value of the underlying objective function
    rho::Float64
    fcval::Vector{Float64} # values of the constraints
    rhoc::Vector{Float64} # rho value for each constraint
    xcur::Vector{Float64}
    gval::Vector{Float64}
    wval::Vector{Float64}
    gcval::Vector{Float64}
end

function dual_function!(m::Int, y::Vector{Float64}, grad::Vector{Float64}, d::dual_data)::Float64
    #=
    In this function, we solve the CCSA subproblem subject to linear and
    separable quadratic approximation... which has a simple closed form
    solution in this case. All we really have to do is check to make sure we
    stay within the simple lower and upper bounds for the problem
    =#

    # m is the dimension of the constraints
    # y is the current y values, which is needed to evaluate the objective



    d.count_dual+=1

    #=
    for our value of gval_i we have that
        gval_i(x) = f_i(x^k) + \grad f_i(x^k) (x - x^k)
                    + \rho_{i,k,l}/2 sum_{j=1}^n square((x_j - x_j^k) / sigma_j^k)
    where sigma^k and x^k are the values of sigma and x in the outer iteration
    and \rho{i,k,l} is the value of rho for the k-th outer iteration and l-th
    inner iteration
    =#

    val = d.fval
    d.wval = 0
    val += vecdot(y, fcval)

    for j = 1:d.n

        # special case for lb[i] == ub[i] and thus sigma = 0, dx = 0
        if d.sigma[j] == 0
            d.xcur[j] = d.x[j]
        end

        u = d.rho
        v = d.dfdx[j]
        u += vecdot(d.rhoc, y)
        v += vecdot(d.dfcdx[:,j], y)

        dx = -sqrt(d.sigma[j]) * v / u

        # Check to see if dx is out of bounds. If dx is out of bounds, by
        # convexity we know the minimum is at the bound on the side of dx
        if (abs(dx) > d.sigma[j])
            d.xcur[j] = copysign(d.sigma[j], dx)
        end
        d.xcur[j] = d.x[j] + dx

        # If we leave the bounds, we must go back into the bounds
        if (d.xcur[j] > d.ub[j])
            d.xcur[j] = d.ub[j]
        elseif (d.xcur[j] < d.lb[j])
            d.xcur[j] = d.lb[j]
        end
        dx = d.xcur[j] - d.x[j]

        val += v*dx + 0.5 * u * dx^2 / sqrt(d.sigma[j])

        # Update gval, wval, gcval (the approximation functions)
        dx2sig =  0.5 * dx^2 / sqrt(d.sigma[j])
        d.gval += d.dfdx[j] * dx + d.rho * dx2sig
        d.wval += dx2sig
        d.gcval += d.dfcdx[:,j] * dx + d.rhoc * dx2sig
    end
    if grad
        d.grad = - d.gcval
    end

    return -val
end





function gfunc(n::Int, f::Float64, dfdx::Vector{Float64}, rho::Float64,
    sigma::Vector{Float64}, x0::Vector{Float64}, x::Vector{Float64}, grad::Vector{Float64})::Float64

    # Computers g(x-x0) and its gradient
    val = f

    sigma2inv = 1.0 ./ sqrt(sigma)
    dx = x - x0
    val += vecdot(d.dfdx, dx)
    val += 0.5 * rho * vecdot(sigma2inc, sqrt(dx))
    if grad
        d.grad = d.dfdx .+ rho .* dx .* sigma2inv
    end

    return val
end


function g0(n::Int, x::Vector{Float64}, grad::Vector{Float64}, d::dual_data)::Float64
    d.count_dual += 1
    return gfunc(n, d.fval, d.dfdx, d.rho, d.sigma, d.x, x, grad)
end


function gi(m::Int, n::Int, x::Vector{Float64}, grad::Vector{Float64}, d::dual_data)::Vector{Float64}
    result = Vector{Float64}(m)
    for i =1:m
        result[i] = gfunc(n, d.fcval[i], d.dfcdx[i,:], d.rhoc[i], d.sigma, d.x, x, grad)
    end
    return result
end


function ccsa_quadratic_minimize(
    n::Int64, f::Function, m::Int64, fc::Function,
    lb::Vector{Float64}, ub::Vector{Float64},
    x::Vector{Float64}
    )::Vector{Float64}

    #=
    n is the number of variables
    f is the function that we are minimizing
    fc is the constraint functions, which map a vector in R^n to R^m
    lb is the lower bound on the variables
    ub is the upper bound on the variables
    x is the initial guess
    we return the value of x that is the optimum... I could improve this later
    =#
end



println("Hello")
