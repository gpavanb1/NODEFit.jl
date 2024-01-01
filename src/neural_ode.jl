using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots

using Lux: Chain

function fit_node(dudt::Lux.Chain, t, ode_data, plot_train=true)
    rng = Random.default_rng()
    u0 = ode_data[:, 1]
    tspan = (t[1], t[end])
    p, st = Lux.setup(rng, dudt)
    prob_neuralode = NeuralODE(dudt, tspan, Tsit5(); saveat=t)

    function predict_neuralode(p)
        Array(prob_neuralode(u0, p, st)[1])
    end

    function loss_neuralode(p)
        pred = predict_neuralode(p)
        loss = sum(abs2, ode_data .- pred)
        return loss, pred
    end

    callback = function (p, l, pred; doplot=plot_train)
        println("Loss: $(l)")
        # plot current prediction against data
        if doplot
            plt = scatter(t, ode_data[1, :]; label="data")
            scatter!(plt, t, pred[1, :]; label="prediction")
            display(plot(plt))
        end
        return false
    end

    pinit = ComponentArray(p)

    # First round of training
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, pinit)
    result_neuralode = Optimization.solve(optprob, OptimizationOptimisers.ADAM(0.05); callback=callback,
        maxiters=200)

    # Second round of training
    optprob2 = remake(optprob; u0=result_neuralode.u)
    result_neuralode2 = Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm=0.01);
        callback=callback, allow_f_increases=false, maxiters=200)

    result_neuralode2
end