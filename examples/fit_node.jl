using NODEFit
using Random
using Lux: Chain, Dense
using OrdinaryDiffEq

ndim, nhid = 2, 4
datasize = 30
u0 = Float32[2.0; 0.0]
tspan = (0.0f0, 1.5f0)
tsteps = collect(Float32, range(tspan[1], tspan[2]; length=datasize))

# Solve true ODE to generate data
function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end
prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
data = Array(solve(prob_trueode, Tsit5(); saveat=tsteps))

neural_net = Chain(x -> x .^ 3, Dense(ndim, nhid, tanh), Dense(nhid, ndim))
result_node = NODEFit.fit_node(neural_net, tsteps, data)