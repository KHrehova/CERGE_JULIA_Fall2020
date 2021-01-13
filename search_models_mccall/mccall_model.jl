## Setup
Pkg.add("InstantiateFromURL")
using InstantiateFromURL
# optionally add arguments to force installation: instantiate = true, precompile = true
github_project("QuantEcon/quantecon-notebooks-julia", version = "0.8.0")

## Here’s the distribution of wage offers we’ll work with
using LinearAlgebra, Statistics
Pkg.resolve()
using Distributions, Expectations, NLsolve, Roots, Random, Plots, Parameters
gr(fmt = :png);
n = 50
dist = BetaBinomial(n, 200, 100) # probability distribution
@show support(dist)
w = range(10.0, 60.0, length = n+1) # linearly space wages

using StatsPlots
plt = plot(w, pdf.(dist, support(dist)), xlabel = "wages", ylabel = "probabilities", legend = false)

## We can explore taking expectations over this distribution:
E = expectation(dist) # expectation operator

# exploring the properties of the operator
wage(i) = w[i+1] # +1 to map from support of 0
E_w = E(wage)
E_w_2 = E(i -> wage(i)^2) - E_w^2 # variance
@show E_w, E_w_2

# use operator with left-multiply
@show E * w # the `w` are values assigned for the discrete states
@show dot(pdf.(dist, support(dist)), w); # identical calculation

## To implement our algorithm, let’s have a look at the sequence of
# approximate value functions that this fixed point algorithm generates.
# Default parameter values are embedded in the function.
# Our initial guess v is the value of accepting at every given wage

# parameters and constant objects

c = 25
β = 0.99
num_plots = 6

# Operator
T(v) = max.(w/(1 - β), c + β * E*v) # (5) broadcasts over the w, fixes the v
# alternatively, T(v) = [max(wval/(1 - β), c + β * E*v) for wval in w]

# fill in  matrix of vs
vs = zeros(n + 1, 6) # data to fill
vs[:, 1] .= w / (1-β) # initial guess of "accept all"

# manually applying operator
for col in 2:num_plots
    v_last = vs[:, col - 1]
    vs[:, col] .= T(v_last)  # apply operator
end
plot(vs)

##One approach to solving the model is to directly implement this sort of iteration, and continues until measured deviation between successive iterates is below tol
function compute_reservation_wage_direct(params; v_iv = collect(w ./(1-β)), max_iter = 500,
                                         tol = 1e-6)
    @unpack c, β, w = params

    # create a closure for the T operator
    T(v) = max.(w/(1 - β), c + β * E*v) # (5) fixing the parameter values

    v = copy(v_iv) # start at initial value.  copy to prevent v_iv modification
    v_next = similar(v)
    i = 0
    error = Inf
    while i < max_iter && error > tol
        v_next .= T(v) # (4)
        error = norm(v_next - v)
        i += 1
        v .= v_next  # copy contents into v.  Also could have used v[:] = v_next
    end
    # now compute the reservation wage
    return (1 - β) * (c + β * E*v) # (2)
end

## In the above, we use v = copy(v_iv) rather than just v_iv = v.
```
To understand why, first recall that v_iv is a function argument – either defaulting to the given value, or passed into the function

If we had gone v = v_iv instead, then it would have simply created a new name v which binds to whatever is located at v_iv.
Since we later use v .= v_next later in the algorithm, the values in it would be modified.
Hence, we would be modifying the v_iv vector we were passed in, which may not be what the caller of the function wanted.
The big issue this creates are “side-effects” where you can call a function and strange things can happen outside of the function that you didn’t expect.
If you intended for the modification to potentially occur, then the Julia style guide says that we should call the function compute_reservation_wage_direct! to make the possible side-effects clear.
As usual, we are better off using a package, which may give a better algorithm and is likely to less error prone.

In this case, we can use the fixedpoint algorithm discussed in our Julia by Example lecture to find the fixed point of the T operator.
```
function compute_reservation_wage(params; v_iv = collect(w ./(1-β)), iterations = 500,
                                  ftol = 1e-6, m = 6)
    @unpack c, β, w = params
    T(v) = max.(w/(1 - β), c + β * E*v) # (5) fixing the parameter values

    v_star = fixedpoint(T, v_iv, iterations = iterations, ftol = ftol,
                        m = 0).zero # (5)
    return (1 - β) * (c + β * E*v_star) # (3)
end
```
Let’s compute the reservation wage at the default parameters
```
mcm = @with_kw (c=25.0, β=0.99, w=w) # named tuples

compute_reservation_wage(mcm()) # call with default parameters

```
Comparative Statics
Now we know how to compute the reservation wage, let’s see how it varies with parameters.

In particular, let’s look at what happens when we change β and c.
```

grid_size = 25
R = rand(grid_size, grid_size)

c_vals = range(10.0, 30.0, length = grid_size)
β_vals = range(0.9, 0.99, length = grid_size)

for (i, c) in enumerate(c_vals)
    for (j, β) in enumerate(β_vals)
        R[i, j] = compute_reservation_wage(mcm(c=c, β=β)) # change from defaults
    end
end

contour(c_vals, β_vals, R',
        title = "Reservation Wage",
        xlabel = "c",
        ylabel = "beta",
        fill = true)
```
        As expected, the reservation wage increases both with patience and with unemployment compensation.

Computing the Optimal Policy: Take 2
The approach to dynamic programming just described is very standard and broadly applicable.

For this particular problem, there’s also an easier way, which circumvents the need to compute the value function.

Let ψ denote the value of not accepting a job in this period but then behaving optimally in all subsequent periods.

That is,

ψ=c+β∑i=1nV(wi)pi(6)
where V is the value function.

By the Bellman equation, we then have

V(wi)=max{wi1−β,ψ}
Substituting this last equation into (6) gives

ψ=c+β∑i=1nmax{wi1−β,ψ}pi(7)
Which we could also write as ψ=Tψ(ψ) for the appropriate operator.

This is a nonlinear equation that we can solve for ψ.

One solution method for this kind of nonlinear equation is iterative.

That is,

Step 1: pick an initial guess ψ.

Step 2: compute the update ψ′ via

ψ′=c+β∑i=1nmax{wi1−β,ψ}pi(8)
Step 3: calculate the deviation |ψ−ψ′|.

Step 4: if the deviation is larger than some fixed tolerance, set ψ=ψ′ and go to step 2, else continue.

Step 5: return ψ.

Once again, one can use the Banach contraction mapping theorem to show that this process always converges.

The big difference here, however, is that we’re iterating on a single number, rather than an n-vector.

Here’s an implementation:
```

function compute_reservation_wage_ψ(c, β; ψ_iv = E * w ./ (1 - β), max_iter = 500,
                                    tol = 1e-5)
    T_ψ(ψ) = [c + β * E*max.((w ./ (1 - β)), ψ[1])] # (7)
    # using vectors since fixedpoint doesn't support scalar
    ψ_star = fixedpoint(T_ψ, [ψ_iv]).zero[1]
    return (1 - β) * ψ_star # (2)
end
compute_reservation_wage_ψ(c, β)

```
You can use this code to solve the exercise below.

Another option is to solve for the root of the Tψ(ψ)−ψ equation
    ```
    function compute_reservation_wage_ψ2(c, β; ψ_iv = E * w ./ (1 - β), max_iter = 500,
                                         tol = 1e-5)
        root_ψ(ψ) = c + β * E*max.((w ./ (1 - β)), ψ) - ψ # (7)
        ψ_star = find_zero(root_ψ, ψ_iv)
        return (1 - β) * ψ_star # (2)
    end
    compute_reservation_wage_ψ2(c, β)

```    Exercises
    Exercise 1
    Compute the average duration of unemployment when β=0.99 and c takes the following values

    c_vals = range(10, 40, length = 25)

    That is, start the agent off as unemployed, computed their reservation wage given the parameters, and then simulate to see how long it takes to accept.

    Repeat a large number of times and take the average.

    Plot mean unemployment duration as a function of c in c_vals.
Check solution here:        https://julia.quantecon.org/dynamic_programming/mccall_model.html
        ```
