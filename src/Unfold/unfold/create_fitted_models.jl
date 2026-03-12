using UnfoldBIDS
using Unfold
using UnfoldMakie
using LazyArtifacts
using CUDA
using CUDA.CUSPARSE
using BSplineKit

using Parquet, DataFrames, Tables

using Printf
using Serialization
using PythonCall
using CairoMakie
using Statistics
using StatsPlots

#=
TODO
- model stimulus onset erp like in paper
=#

include(joinpath(@__DIR__, "processed_subs_symlink_setup.jl"))
const py_mne = pyimport("mne")

function mne_raw_to_julia(raw_mne)
    eeg_data = pyconvert(Array{Float64}, raw_mne.get_data(picks = "eeg")) .* 1e6
    sfreq = pyconvert(Float64, raw_mne.info["sfreq"])

    # pandas df to julia
    ann_pd = raw_mne.annotations.to_data_frame()
    ann_dict = ann_pd.to_dict("list")
    ann_keys = pyconvert(Vector{String}, pybuiltins.list(ann_dict.keys()))
    ann_pairs = map(k -> Symbol(k) => pyconvert(Vector, ann_dict[k]), ann_keys)
    ann_df = DataFrame(ann_pairs)
    ann_df.onset = pyconvert(Vector{Float64}, raw_mne.annotations.onset)

    return eeg_data, sfreq, ann_df
end

# Workaround for Unfold GPU-QR path with current CUDA sparse matrices:
# Unfold.prepare_XTX currently tries CuArray{T,...}(CuSparseMatrixCSC), which can
# trigger scalar-indexing errors. Convert via CuArray(...) explicitly.
import Unfold: prepare_XTX
function Unfold.prepare_XTX(
    Ĥ,
    data::CuArray{T},
    X::CuSparseMatrixCSC{T2},
) where {T,T2}
    Xt = X'
    R_xx = CuArray(Xt * X)
    R_xy = similar(data, size(X, 2))
    return Ĥ, data, (Xt, R_xx, R_xy)
end

#bids_root  = "/pfs/work9/workspace/scratch/st_st156392-mydata/mergedDataset"
#bids_root  = "/home/oki/storage/mergedDataset"
#bids_root = "/home/oki/ehlers-work2/mergedDataset"
bids_root = "/home/oki/ehlers-work2/mergedDataset"

deriv_root = joinpath(bids_root, "derivatives")
analyzed_subjects = ["NDARAG788YV9","NDARAP457WB5","NDARAY298THW","NDARAY461TZZ","NDARBT607PZL","NDARCT889DMB","NDARCW933FD5","NDARDK794WV3","NDARDL033XRG","NDAREG013BLG","NDARFK819TD5","NDARGA890MKA","NDARGH074MU6","NDARGN210CK7","NDARGN483WFH","NDARGU271CPG","NDARGV455JV1","NDARGX760NYV","NDARGY559UL3","NDARHE896MYM","NDARHP841RMR","NDARHT518WEM","NDARJT064LRE","NDARKB712GAP","NDARKG016KD1","NDARKH291KRE","NDARKH837TB2","NDARKT312RUD","NDARKX665ZD3","NDARKY667THK","NDARLP413TUX","NDARLY687YDQ","NDARMA390CHB","NDARMC950EUL","NDARMF116AFR","NDARNJ633HHX","NDARPD568LHV","NDARPG873DJP","NDARPT417LW6","NDARPW746FWF","NDARRK135YAZ","NDARRK528GFZ","NDARRL379BET","NDARRU751ATE","NDARTA920XFC","NDARTB883GUN","NDARTJ862ENU","NDARUC804LKP","NDARUK054GTN","NDARUM009GEZ","NDARVB819ENX","NDARVD609JNZ","NDARWB782FLR","NDARWT403LP6","NDARWV403LV8","NDARXF860CZ7","NDARXR865BVX","NDARXY240WJC","NDARXZ692ULW","NDARYA857NDW","NDARYE221LZB","NDARYJ413BLN","NDARZG044CJ5","NDARZT581RNV"]

task = "allTasks"
run_id = "1"
model_dump_dir = joinpath(@__DIR__, "uf2_model_dumps")
mkpath(model_dump_dir)

# find subjects that have a folder inside derivatives
processed_subs = filter(name -> startswith(name, "sub-") && isdir(joinpath(deriv_root, name)), readdir(deriv_root))
@info "Found $(length(processed_subs)) processed subjects in derivatives" processed_subs


#only need to run once
#setup_processed_subs_symlinks(
#    processed_subs,
#    bids_root,
#    deriv_root;
#    task = task,
#    run_id = run_id,
#)


layout_df = bids_layout(bids_root, derivatives = true, task = task, run = run_id)

function fit_subject_model(subject::AbstractString, layout_df::DataFrame)

    # Only use proc-clean_raw_eeg.fif (not epoched, cleaned with ICA)
    df_subject = filter(row ->
        endswith(row.file, "proc-clean_raw_eeg.fif") &&
        row.subject == subject,
        layout_df
    )
    @assert !isempty(df_subject) "No cleaned raw EEG file found for subject $subject"

    data_df = load_bids_eeg_data(
        df_subject;
        loading_function = file_path -> py_mne.io.read_raw_fif(file_path, verbose = "ERROR"),
    )
    @assert nrow(data_df) > 0 "Could not load EEG data for subject $subject"

    raw_mne = data_df.raw[1]
    eeg_data, sfreq, ann_df = mne_raw_to_julia(raw_mne)

    # only use left-eye saccades and fixations
    filter!(:description => x -> x in ("ET_Fixation L", "ET_Saccade L"), ann_df)
    sort!(ann_df, :onset)

    descs = ann_df.description
    # For FRPs at fixation onset, use the incoming (preceding) saccade amplitude.
    fix_idx = findall(i ->
        descs[i] == "ET_Fixation L" &&
        i > firstindex(descs) &&
        descs[i - 1] == "ET_Saccade L",
        eachindex(descs)
    )

    @assert !isempty(fix_idx) "No fixation events with direct preceding saccade for subject $subject"
    prev_idx = fix_idx .- 1

    skipped_fix = count(==("ET_Fixation L"), descs) - length(fix_idx)
    if skipped_fix > 0
        @warn "Dropping fixation events without direct preceding saccade (maybe dropped because of blink)" subject skipped_fix
    end

    merged_df = DataFrame(
        latency        = round.(Int, 1 .+ ann_df.onset[fix_idx] .* sfreq),
        type           = fill("fixation", length(fix_idx)),
        onset_sec      = ann_df.onset[fix_idx],
        fix_duration   = ann_df.duration[fix_idx],
        sacc_amplitude = ann_df[prev_idx, "Amplitude"],
    )

    # copy other columns from the fixation rows.
    other_cols = setdiff(names(ann_df), ["onset", "duration", "description", "Amplitude"])
    for col in other_cols
        merged_df[!, col] = ann_df[fix_idx, col]
    end

    fix = deepcopy(merged_df)
    fix.type = fill("fixation", nrow(fix))

    select!(fix, [:latency, :sacc_amplitude, Symbol("Location X"), Symbol("Location Y")])
    rename!(fix, Symbol("Location X") => :fixation_position_x, Symbol("Location Y") => :fixation_position_y)

    min_sacc_amp_deg = 0.5
    q90 = quantile(collect(skipmissing(fix.sacc_amplitude)), 0.90)
    fix_top90 = filter(
        :sacc_amplitude => x -> !ismissing(x) && x >= min_sacc_amp_deg && x < q90,
        fix,
    )

    #histogram(
    #    skipmissing(fix_top90.sacc_amplitude);
    #    bins=40,
    #    xlabel="sacc_amplitude",
    #    ylabel="Count",
    #    title="Distribution of sacc_amplitude",
    #    legend=false
    #)

    basis_fix = firbasis(τ = (-0.2, 0.6), sfreq = sfreq)

    #f_fix = @formula(
    #    0 ~ 1 +
    #        spl(fixation_position_x, 5) +
    #        spl(fixation_position_y, 5) +
    #        spl(sacc_amplitude, 5)
    #)

    f_fix = @formula(0 ~ 1)
    
    design = [Any => (f_fix, basis_fix)]
    gpu_solver = (x, y) -> Unfold.solver_predefined(x, y; solver = :qr)

    #takes 0:00:21
    return Unfold.fit(UnfoldModel, design, fix_top90, CUDA.cu(eeg_data), solver = gpu_solver)

    #takes 0:03:14
    #uf = fit(UnfoldModel, design, fix_top90, eeg_data)
end

saved_subjects = String[]
failed_subjects = NamedTuple{(:subject, :error),Tuple{String,String}}[]

for subject in analyzed_subjects
    @info "Fitting model for subject" subject
    try
        model = fit_subject_model(subject, layout_df)
        model_file = joinpath(model_dump_dir, "model_sub-$(subject).jls")
        open(model_file, "w") do io
            serialize(io, model)
        end
        push!(saved_subjects, subject)
        @info "Saved fitted model" subject model_file
    catch err
        err_msg = sprint(showerror, err)
        push!(failed_subjects, (subject = subject, error = err_msg))
        @warn "Skipping subject because model fitting failed" subject error = err_msg
    end
end

@info "Finished subject-level fitting" requested = length(analyzed_subjects) fitted = length(saved_subjects) failed = length(failed_subjects) model_dump_dir
if !isempty(failed_subjects)
    @info "Subjects skipped due to fitting errors" failed_subjects
end
