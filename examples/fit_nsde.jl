using NODEFit
using Random
using Lux: Chain, Dense
using StochasticDiffEq
using SciMLBase.EnsembleAnalysis

ndim, nhid = 2, 4
datasize = 30
u0 = Float32[2.0; 0.0]
tspan = (0.0f0, 1.5f0)
tsteps = collect(Float32, range(tspan[1], tspan[2]; length=datasize))

# Define the networks
drift_net = Chain(x -> x .^ 3, Dense(ndim, nhid, tanh), Dense(nhid, ndim))
diff_net = Dense(ndim, ndim)

# Define the sample SDE problem
# Drift
function trueSDEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end
# Diffusion
mp = Float32[0.3, 0.3]
function true_noise_func(du, u, p, t)
    du .= mp .* u
end
# Problem
prob_truesde = SDEProblem(trueSDEfunc, true_noise_func, u0, tspan)
println("Solving initial SDE...")
# Take a typical sample from the mean
ensemble_prob = EnsembleProblem(prob_truesde; safetycopy=false)
ensemble_sol = solve(ensemble_prob, SOSRI(); trajectories=2)
sde_data, sde_data_vars = Array.(timeseries_point_meanvar(ensemble_sol, tsteps))
println("Solved")

result_nsde = NODEFit.fit_nsde(drift_net, diff_net, tsteps, sde_data, sde_data_vars)