using UnfoldBIDS
using Unfold
using UnfoldMakie
using BSplineKit
using CairoMakie

using DataFrames
using PythonCall

const py_mne = pyimport("mne")
model_dump_dir = "./uf_models"
mkpath(model_dump_dir)

function mne_raw_to_julia(raw_mne)
    #don't fit on et channels.
    eeg_data = pyconvert(Array{Float64}, raw_mne.get_data(picks = "eeg", reject_by_annotation = "omit")) .* 1e6
    sfreq = pyconvert(Float64, raw_mne.info["sfreq"])

    ann_pd = raw_mne.annotations.to_data_frame()
    ann_dict = ann_pd.to_dict("list")
    ann_keys = pyconvert(Vector{String}, pybuiltins.list(ann_dict.keys()))
    ann_pairs = map(k -> Symbol(k) => pyconvert(Vector, ann_dict[k]), ann_keys)
    ann_df = DataFrame(ann_pairs)
    ann_df.onset = pyconvert(Vector{Float64}, raw_mne.annotations.onset)

    return eeg_data, sfreq, ann_df
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

    # keep only fixations with preceding saccade amplitude >= 0.3 deg,
    # and winsorize amplitudes above 15 deg
    min_sacc_amp_deg = 0.3
    max_sacc_amp_deg = 15.0
    fix_filt = filter(:sacc_amplitude => x -> !ismissing(x) && x >= min_sacc_amp_deg, fix)
    amps = Float64.(fix_filt.sacc_amplitude)
    amps = min.(amps, max_sacc_amp_deg)
    fix_filt.sacc_amplitude = amps

    ## winsorize amplitudes < 0.5 and top 10%
    #min_sacc_amp_deg = 0.5
    #fix_filt = filter(:sacc_amplitude => x -> !ismissing(x), fix)
    #amps = Float64.(fix_filt.sacc_amplitude)
    #amps = max.(amps, min_sacc_amp_deg)
    #n = length(amps)
    #if n > 1
    #    n_top = clamp(ceil(Int, 0.10 * n), 1, n - 1)
    #    cutoff = sort(amps)[n - n_top]
    #    amps = min.(amps, cutoff)
    #end
    #fix_filt.sacc_amplitude = amps

    return fix_filt
end


function fit_single_file_model(fif_path::AbstractString)
    raw_mne = py_mne.io.read_raw_fif(fif_path, preload = true)
    eeg_data, sfreq, ann_df = mne_raw_to_julia(raw_mne)
    fix_filt = prepare_fixation_events(ann_df, sfreq)

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
    return model
end

function init_models()
    bids_root = "/home/oki/ehlers-work2/mergedDataset"
    task = "ThePresent"
    run_id = "1"
    file_suffix = "proc-eyelink_raw" #proc-clean_raw"

    layout_df = bids_layout(bids_root; derivatives = true, task = task, run = run_id)
    subject_files = filter(row -> endswith(row.file, file_suffix * ".fif"), layout_df)
    subject_files = sort(subject_files, [:subject, :file])

    @info "Found candidate FIF files for fitting" n_files = nrow(subject_files)

    for row in eachrow(subject_files)
        subject = String(row.subject)
        fif_path = String(row.file)

        @info "Fitting model for subject" subject fif_path
        try
            model = fit_single_file_model(fif_path)
            model_file = joinpath(
                model_dump_dir,
                "$(subject)|$(task)|$(run_id)|$(file_suffix).jld2",
            )
            Unfold.save(model_file, model; compress = false)
            @info "Saved fitted model" subject model_file
        catch err
            @warn "Skipping subject because model fitting failed" subject error = sprint(showerror, err)
        end
    end
end

########################### Analysis

function plot_eff(
    subject_ids::AbstractVector,
    run_id,
    task_id::AbstractString,
    model_type::AbstractString,
    sacc_amplitude_bins,
    electrode::AbstractString,
)
    electrode_num = try
        parse(Int, replace(uppercase(strip(electrode)), "E" => ""))
    catch
        error("Could not parse electrode '$electrode'. Expected format like \"E82\".")
    end

    run_id_str = string(run_id)
    eff_list = DataFrame[]

    for subject in subject_ids
        subject_str = String(subject)
        model_file = joinpath(
            model_dump_dir,
            "$(subject_str)|$(task_id)|$(run_id_str)|$(model_type).jld2",
        )

        if !isfile(model_file)
            @warn "Model file not found, skipping subject" subject = subject_str model_file
            continue
        end

        loaded = Unfold.load(model_file)
        model =
            if loaded isa AbstractDict
                if haskey(loaded, "uf")
                    @info "key here"
                    loaded["uf"]
                elseif haskey(loaded, :uf)
                    @info "key there"
                    loaded[:uf]
                else
                    error("Loaded file does not contain key \"uf\": $model_file")
                end
            else
                loaded
            end

        eff = effects(Dict(:sacc_amplitude => sacc_amplitude_bins), model)
        eff_electrode = subset(
            eff,
            :channel => ByRow(==(electrode_num)),
            :yhat => ByRow(!ismissing),
        )

        if nrow(eff_electrode) == 0
            @warn "No effects rows for electrode, skipping subject" subject = subject_str electrode
            continue
        end

        eff_electrode.subject = fill(subject_str, nrow(eff_electrode))
        push!(eff_list, eff_electrode)
    end

    isempty(eff_list) && error("No effects data available for the requested inputs.")

    eff_all = vcat(eff_list...)

    eff_plot =
        if length(eff_list) == 1
            select(eff_all, Not(:subject))
        else
            group_cols = filter(c -> c ∉ [:yhat, :subject], names(eff_all))
            combine(groupby(eff_all, group_cols), :yhat => mean => :yhat)
        end

    fig = plot_erp(
        eff_plot;
        mapping = (; y = :yhat, color = :sacc_amplitude, group = :sacc_amplitude),
    )
    display(fig)
    return fig, eff_plot
end

plot_eff(["NDARFT305CG1"], 1, "ThePresent", "proc-clean_raw", 1:3:12, "E8")