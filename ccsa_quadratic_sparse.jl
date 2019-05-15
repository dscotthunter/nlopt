#=
In this file I implement Svanberg's CCSA algorithm with simple linear
approximation + quadratic penalty function, where I also make use of
Julia's sparse array package for the Jacobian of the constraints. This
implementation is based on Steven Johnson's nlopt implementation of
ccsa_quadratic. I only implement the non-preconditioned case based on the
Svanberg paper! I would also like to thank Steven Johnson for recommending
that I take advantage of the Julia sparse array package.
=#

using LinearAlgebra, SparseArrays, NLopt

# Basing this struct off of Steven Johnson's implementation
mutable struct dual_data
    count::Int # number of total inner + outer approximations
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
    rhoc::Vector{Float64} # rho value for each constraint so m
    xcur::Vector{Float64}
    gval::Float64
    wval::Float64
    gcval::Vector{Float64}
end

function dual_function!(y::Vector{Float64}, grad::Vector{Float64}, d::dual_data)::Float64
    #=
    In this function, we solve the CCSA subproblem subject to linear and
    separable quadratic approximation... which has a simple closed form
    solution in this case. All we really have to do is check to make sure we
    stay within the simple lower and upper bounds for the problem
    =#


    # y is the current y values, which is needed to evaluate the objective

    m = length(y)


    d.count+=1

    #=
    for our value of gval_i we have that
        gval_i(x) = f_i(x^k) + \grad f_i(x^k) (x - x^k)
                    + \rho_{i,k,l}/2 sum_{j=1}^n square((x_j - x_j^k) / sigma_j^k)
    where sigma^k and x^k are the values of sigma and x in the outer iteration
    and \rho{i,k,l} is the value of rho for the k-th outer iteration and l-th
    inner iteration
    =#
    d.gval = d.fval
    val = d.fval
    d.wval = 0.0
    val += dot(y, d.fcval)
    for i=1:m
        d.gcval[i] = d.fcval[i]
    end

    for j = 1:d.n
        # special case for lb[i] == ub[i] and thus sigma = 0, dx = 0
        if d.sigma[j] == 0
            d.xcur[j] = d.x[j]
            continue
        end

        u = d.rho
        v = d.dfdx[j]
        u += dot(d.rhoc, y)
        v += dot(d.dfcdx[:,j], y)

        dx = -sqrt(d.sigma[j]) * v / u

        # Check to see if dx is out of bounds. If dx is out of bounds, by
        # convexity we know the minimum is at the bound on the side of dx
        if (abs(dx) > d.sigma[j])
            dx = copysign(d.sigma[j], dx)
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
        for i=1:m
            d.gcval[i] += d.dfcdx[i,j] * dx + d.rhoc[i] * dx2sig
        end
    end
    if length(grad)>0
        for i=1:m
            grad[i] = - d.gcval[i]
        end
    end

    return -val
end





function gfunc(n::Int, f::Float64, dfdx::Vector{Float64}, rho::Float64,
    sigma::Vector{Float64}, x0::Vector{Float64}, x::Vector{Float64}, grad::Vector{Float64})::Float64

    # Computes g(x-x0) and its gradient
    val = f

    sigma2inv = 1.0 ./ sqrt(sigma)
    dx = x - x0
    val += dot(dfdx, dx)
    val += 0.5 * rho * dot(sigma2inv, sqrt(dx))
    if length(grad) > 0
        for i=1:n
            grad[i] = dfdx[i] + rho * dx[i] * sigma2inv
        end
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

function nlopt_stop_ftol(vold::Float64, vnew::Float64)::Bool
    reltol = 0.0
    abstol = 0.0
    return (abs(vnew - vold) < abstol || abs(vnew - vold) < reltol * (abs(vnew) + abs(vold)) * 0.5 || (reltol > 0 && vnew == vold))
end

function nlopt_stop_x(oldx::Vector{Float64}, x::Vector{Float64})::Bool
    xtol_rel = 0.0
    xtol_abs = 0.0
    if (norm(x - oldx) < xtol_rel * norm(x))
        return true
    end
    for i=1:length(x)
        if (abs(x[i] - oldx[i]) >= xtol_abs)
            return false
        end
    end
    return true
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
    # Some user defined variables for the code. RHOMIN is the minimum value of
    # rho, which I took from the paper.
    ccsa_verbose = false
    CCSA_RHOMIN = 1e-5

    dfdx_cur = Vector{Float64}(undef, n)
    fcur = f(x, dfdx_cur)

    ### update me
    dfcdx_cur = spzeros(m,n)
    fcval_cur = fc(x, dfcdx_cur)
    dd = dual_data(0, n, deepcopy(x), lb, ub, Vector{Float64}(undef, n), dfdx_cur, dfcdx_cur, fcur, 1.0, fcval_cur, ones(Float64, m), deepcopy(x), 0.0, 0.0, Vector{Float64}(undef, n) )
    dual_lb = zeros(Float64, m)
    dual_ub = 10.0^40 * ones(Float64, m)
    y = zeros(Float64, m)



    rho = 1.0
    minf = fcur
    k = 0



    for i=1:n
        if (isinf(ub[i]) || isinf(lb[i]))
            dd.sigma[i] = 1.0
        else
            dd.sigma[i] = 0.5 * (ub[i] - lb[i])
        end
    end

    feasible=true
    infeasibility=0.0
    for i=1:m
        feasible = feasible && (fcval_cur[i] <= 0.0)
        if (fcval_cur[i] > infeasibility)
            infeasibility = fcval_cur[i]
        end
    end

    if (!feasible)
        dual_ub = 1.0e40 * ones(Float64, m)
    end
    xprev = deepcopy(dd.xcur)
    while true # outer iterations of the CCSA algorithm
        fprev = fcur
        k+=1
        if k>1
            xprevprev = deepcopy(xprev)
        end
        xprev = deepcopy(dd.xcur)
        while true # inner iterations of the CCSA algorithm
            # Solve the dual problem
            dd.rho = rho
            dd.count = 0
            if ccsa_verbose
                println("Minimizing dual in ccsa")
            end
            opt = Opt(:LD_CCSAQ, m)
            opt.lower_bounds = dual_lb
            opt.upper_bounds = dual_ub
            opt.ftol_rel = 1e-14
            opt.ftol_abs = 0.0
            opt.maxeval = 100000
            min_objective!(opt, (x,grad) -> dual_function!(x,grad,dd))
            (minfy,y,ret) = optimize(opt, y)


            # Evaluate the dual function
            if ccsa_verbose
                println("CCSA dual converged in $(dd.count) iterations to g = $(dd.gval) \n")
            end

            fcur = f(dd.xcur, dfdx_cur)

            inner_done = (dd.gval >= fcur)

            infeasibility_cur = 0.0
            feasible_cur = true

            # Check feasibility and see if we satisfy inner done
            fcval_cur = fc(dd.xcur, dfcdx_cur)
            for i=1:m
                feasible_cur = (feasible_cur && fcval_cur[i] <= 0.0)
                inner_done = inner_done && (dd.gcval[i] >= fcval_cur[i])
                if fcval_cur[i] > infeasibility_cur
                    infeasibility_cur = fcval_cur[i]
                end
            end


            if ((fcur < minf && (inner_done || feasible_cur || !feasible)) || (!feasible && infeasibility_cur < infeasibility))
                if !feasible_cur && ccsa_verbose
                    println("CCSA - using infeasible point?")
                end
                dd.fval =  fcur
                minf = fcur
                infeasibility = infeasibility_cur
                dd.fcval = deepcopy(fcval_cur)
                dd.x = deepcopy(dd.xcur)
                x = deepcopy(dd.x)
                dd.dfdx = deepcopy(dfdx_cur)
                dd.dfcdx = deepcopy(dfcdx_cur)
                if (infeasibility_cur == 0.0)
                    if !feasible
                        dual_ub = 1.0e40 * ones(Float64, m)
                    end
                    feasible = true
                end
            end


            if inner_done
                break
            end

            if fcur > dd.gval
                rho = min(10*rho, 1.1 * (rho + (fcur-dd.gval) / dd.wval))
                dd.rho = rho
            end

            for i=1:m
                if fcval_cur[i] > dd.gcval[i]
                    dd.rhoc[i] = min(10*dd.rhoc[i], 1.1 * (dd.rhoc[i] + (fcval_cur[i]-dd.gcval[i]) / dd.wval))
                end
            end


            if ccsa_verbose
                println("CCSA inner iteration: rho -> $(rho)\n")
                for i=1:m
                    println("       CCSA rhoc[$i] -> $(dd.rhoc[i])")
                end
            end
        end
        # Check convergence criteria
        # nlopt_stop_ftol and nlopt_stop_x
        if nlopt_stop_ftol(fprev, fcur)
            if ccsa_verbose
                println("Stopping ftol reached")
            end
            return dd.xcur
        end

        if nlopt_stop_x(xprev, dd.xcur)
            if ccsa_verbose
                println("Stopping xtol reached")
            end
            return dd.xcur
        end
        # Within .1% for a given dataset -- used for testing
        # if (fcur < 59.843537754)
        #     return dd.xcur
        # end
        dd.x = deepcopy(dd.xcur)

        # Done with inner iteration so updating things for outer iteration
        rho = max(0.1*rho, CCSA_RHOMIN)
        for i=1:m
            dd.rhoc[i] = max(0.1*dd.rhoc[i], CCSA_RHOMIN)
        end
        if ccsa_verbose
            println("CCSA outer iteration: rho -> $(rho)\n")
            for i=1:m
                println("       CCSA rhoc[$i] -> $(dd.rhoc[i])")
            end
        end


        if k>1
            for j=1:n
                dx2 = (dd.xcur[j]-xprev[j]) * (xprev[j]-xprevprev[j])
                gam = dx2 < 0 ? 0.7 : (dx2 > 0 ? 1.2 : 1)
                dd.sigma[j] *= gam
                if !(isinf(dd.ub[j]) || isinf(dd.lb[j]))
                    dd.sigma[j] = min(sigma[j], 10.0 * (ub[j]-lb[j]))
                    # use a smaller lower bound than Svanberg's
                    # 0.01*(ub-lb), which seems unnecessarily large
                    dd.sigma[j] = max(sigma[j], 1e-8 * (ub[j]-lb[j]))
                end
                if ccsa_verbose
                    println("       CCSA sigma[$j] -> $(dd.sigma[j])")
                end
            end
        end
        if ccsa_verbose
            println("x : $(dd.x)\n")
            println("minf : $(minf)\n")
        end
    end
    return dd.xcur
end
