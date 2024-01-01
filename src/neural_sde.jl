using Plots, Statistics, ComponentArrays, Optimization,
    OptimizationOptimisers, DiffEqFlux, StochasticDiffEq, Random
using Lux

function fit_nsde(drift_dudt::Lux.Chain, diffusion_dudt::Lux.Dense, t, sde_data, sde_data_vars, n_trajectories=2, plot_train=true)
    u0 = sde_data[:, 1]
    tspan = (t[1], t[end])

    neuralsde = NeuralDSDE(drift_dudt, diffusion_dudt, tspan, SOSRI();
        saveat=t, reltol=1e-1, abstol=1e-1)
    ps, st = Lux.setup(Random.default_rng(), neuralsde)
    ps = ComponentArray(ps)

    # Fit NeuralSDE
    neuralsde_model = Lux.Experimental.StatefulLuxLayer(neuralsde, nothing, st)

    function predict_neuralsde(p, u=u0)
        return Array(neuralsde_model(u, p))
    end

    function loss_neuralsde(p; n=n_trajectories)
        u = repeat(reshape(u0, :, 1), 1, n)
        samples = predict_neuralsde(p, u)
        means = mean(samples; dims=2)
        vars = var(samples; dims=2, mean=means)[:, 1, :]
        means = means[:, 1, :]
        loss = sum(abs2, sde_data - means) + sum(abs2, sde_data_vars - vars)
        return loss, means, vars
    end

    # Callback function to observe training
    callback = function (p, loss, means, vars; doplot=plot_train)
        # loss against current data
        display(loss)

        # plot current prediction against data
        plt = Plots.scatter(t, sde_data[1, :]; yerror=sde_data_vars[1, :],
            ylim=(-4.0, 8.0), label="data")
        Plots.scatter!(plt, t, means[1, :]; yerror=vars[1, :], label="prediction")
        # push!(list_plots, plt)

        if doplot
            display(plt)
        end
        return false
    end

    # Train NeuralSDE
    opt = Adam(0.025)

    # First round of training with n = 2
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_neuralsde(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, ps)
    result_neuralsde = Optimization.solve(optprob, opt; callback, maxiters=500)

    result_neuralsde
end