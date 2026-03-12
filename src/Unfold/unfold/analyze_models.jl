using Unfold
using UnfoldMakie
using BSplineKit
using CUDA
using Serialization
using StatsPlots
using DataFrames
using CairoMakie
using Statistics

const DEFAULT_MODEL_DUMP_DIR = joinpath(@__DIR__, "uf2_model_dumps")
const SUBJECT = "NDARZG044CJ5"

# electrodes at back of the head
back = [81, 88, 73, 82, 74, 94, 89, 83, 75, 70, 69, 68]

model_file = joinpath(DEFAULT_MODEL_DUMP_DIR, "model_sub-$(SUBJECT).jls")

function plot_saccade_amplitude_histogram(model)
    sacc_amplitude = Float64[]
    for dm in model.designmatrix
        events = dm.events
        if hasproperty(events, :sacc_amplitude)
            append!(sacc_amplitude, collect(skipmissing(events.sacc_amplitude)))
        end
    end

    hist = histogram(
        sacc_amplitude;
        bins = 40,
        xlabel = "sacc_amplitude",
        ylabel = "Count",
        title = "Distribution of sacc_amplitude",
        legend = false,
    )
    display(hist)
    return nothing
end

model = open(model_file, "r") do io
    deserialize(io)
end

println("Loaded model: $(abspath(model_file))")
ct = coeftable(model)
display(ct)

################

plot_saccade_amplitude_histogram(model)

################

#ct_back = subset(ct, :channel => ByRow(in(back)))

ct_back = subset(
    ct,
    :channel => ByRow(in(back)),
    :coefname => ByRow(c -> startswith(c, "spl(sacc_amplitude,"))
)

ct_back_avg = combine(
    groupby(ct_back, [:coefname, :eventname, :group, :time]),
    :estimate => mean => :estimate
)

fig_ct_back = plot_erp(ct_back_avg)
resize!(fig_ct_back.scene, 1400, 700)
#display(fig_ct_back)

################

eff = effects(Dict(:sacc_amplitude => 1:3:12), model)

eff_roi = subset(eff, :channel => ByRow(in(back)))
eff_roi = combine(
    groupby(eff_roi, [:time, :sacc_amplitude]),
    :yhat => mean => :yhat
)

fig = plot_erp(
    eff_roi;
    mapping = (; y = :yhat, color = :sacc_amplitude, group = :sacc_amplitude)
)
resize!(fig.scene, 1400, 700)

################

#fig = plot_erp(eff; mapping = (; color = :sacc_amplitude, group = :sacc_amplitude))



