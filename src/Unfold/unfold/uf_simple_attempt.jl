using Unfold
using UnfoldMakie
using BSplineKit
using CairoMakie

using DataFrames
using PythonCall

const py_mne = pyimport("mne")

function mne_raw_to_julia(raw_mne)
    #dont fit on et channels
    eeg_data = pyconvert(Array{Float64}, raw_mne.get_data(picks = "eeg", reject_by_annotation = "omit")) .* 1e6
    eeg_ch_names = pyconvert(Vector{String}, raw_mne.copy().pick("eeg").ch_names)
    sfreq = pyconvert(Float64, raw_mne.info["sfreq"])

    ann_pd = raw_mne.annotations.to_data_frame()
    ann_dict = ann_pd.to_dict("list")
    ann_keys = pyconvert(Vector{String}, pybuiltins.list(ann_dict.keys()))
    ann_pairs = map(k -> Symbol(k) => pyconvert(Vector, ann_dict[k]), ann_keys)
    ann_df = DataFrame(ann_pairs)
    ann_df.onset = pyconvert(Vector{Float64}, raw_mne.annotations.onset)

    return eeg_data, eeg_ch_names, sfreq, ann_df
end

function prepare_fixation_events(ann_df::DataFrame, sfreq::Real)
    ann_sorted = sort(copy(ann_df), :onset)

    # use fixation onsets with saccade amplitude from preceding saccade
    # only use fixations where saccade was directly before (e.g. no blink inbetween)
    descs = ann_sorted.description
    fix_idx = findall(i ->
        !ismissing(descs[i]) &&
        descs[i] == "ET_Fixation L" &&
        i > firstindex(descs) &&
        !ismissing(descs[i - 1]) &&
        descs[i - 1] == "ET_Saccade L",
        eachindex(descs)
    )

    prev_idx = fix_idx .- 1

    fix = DataFrame(
        latency = round.(Int, 1 .+ ann_sorted.onset[fix_idx] .* sfreq),
        sacc_amplitude = ann_sorted[prev_idx, "Amplitude"],
        fixation_position_x = ann_sorted[fix_idx, "Location X"],
        fixation_position_y = ann_sorted[fix_idx, "Location Y"],
    )

    # winsorize amplitudes < 0.5 and top 10%
    min_sacc_amp_deg = 0.5
    fix_filt = filter(:sacc_amplitude => x -> !ismissing(x), fix)
    amps = Float64.(fix_filt.sacc_amplitude)
    amps = max.(amps, min_sacc_amp_deg)
    n = length(amps)
    if n > 1
        n_top = clamp(ceil(Int, 0.10 * n), 1, n - 1)
        cutoff = sort(amps)[n - n_top]
        amps = min.(amps, cutoff)
    end
    fix_filt.sacc_amplitude = amps

    return fix_filt
end


function fit_single_file_model(fif_path::AbstractString)
    raw_mne = py_mne.io.read_raw_fif(fif_path, preload = true)
    eeg_data, eeg_ch_names, sfreq, ann_df = mne_raw_to_julia(raw_mne)
    fix_filt = prepare_fixation_events(ann_df, sfreq)
    e_idx = findfirst(==(electrode), eeg_ch_names)

    @info "Prepared events" n_events = nrow(fix_filt)

    basis_fix = firbasis(τ = (-0.1, 0.5), sfreq = sfreq)
    f_fix = @formula(
        0 ~ 1 +
            #spl(fixation_position_x, 5) +
            #spl(fixation_position_y, 5) +
            spl(sacc_amplitude, 5)
    )
    design = [Any => (f_fix, basis_fix)]
    solver_fn = (x, y) -> Unfold.solver_predefined(x, y; solver = :qr)
    model = Unfold.fit(UnfoldModel, design, fix_filt, eeg_data, solver = solver_fn)
    return model, e_idx
end



fif_path = "/home/oki/ehlers-work2/mergedDataset/derivatives/sub-NDAREF624KJN/eeg/sub-NDAREF624KJN_task-ThePresent_run-1_proc-filt_raw.fif"

electrode = "E82"

model, e_idx = fit_single_file_model(fif_path)

eff = effects(Dict(:sacc_amplitude => 1:3:12), model)
eff_e82 = subset(eff, :channel => ByRow(==(e_idx)), :yhat => ByRow(!ismissing))
fig = plot_erp(
    eff_e82;
    mapping = (; y = :yhat, color = :sacc_amplitude, group = :sacc_amplitude)
)
mkpath("./processed")
save("./processed/eff_plot.png", fig)
display(fig)
