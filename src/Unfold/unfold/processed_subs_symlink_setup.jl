function is_symlink_to(path::AbstractString, expected_target::AbstractString)
    if !islink(path)
        return false
    end
    target = readlink(path)
    target_abs = isabspath(target) ? normpath(target) : normpath(joinpath(dirname(path), target))
    expected_abs = normpath(String(expected_target))
    return target_abs == expected_abs
end

function setup_processed_subs_symlinks(
    processed_subs,
    bids_root::AbstractString,
    deriv_root::AbstractString;
    task::AbstractString = "freeView",
    run_id::AbstractString = "1",
)
    for sub in processed_subs
        raw_eeg_dir   = joinpath(bids_root,  sub, "eeg")
        deriv_eeg_dir = joinpath(deriv_root, sub, "eeg")

        if !isdir(deriv_eeg_dir)
            @warn "No derivatives eeg dir for $sub, skipping" deriv_eeg_dir
            continue
        end
        if !isdir(raw_eeg_dir)
            @warn "No raw eeg dir for $sub, skipping symlink creation" raw_eeg_dir
            continue
        end

        # 1) Raw events file for this subject and run.
        raw_events_name = "$(sub)_task-$(task)_run-$(run_id)_events.tsv"
        raw_events_path = joinpath(raw_eeg_dir, raw_events_name)

        if !isfile(raw_events_path)
            @warn "Raw events file not found for $sub, skipping" raw_events_path
            continue
        end

        # 2) Process all .fif files with the selected run in the name.
        fif_files = filter(fname ->
            occursin("run-$(run_id)", fname) && endswith(lowercase(fname), ".fif"),
            readdir(deriv_eeg_dir)
        )

        if isempty(fif_files)
            @info "No .fif files with run-$(run_id) for $sub in derivatives" deriv_eeg_dir
            continue
        end

        for fname in fif_files
            old_path = joinpath(deriv_eeg_dir, fname)
            new_fname = fname
            new_path = old_path

            # If needed, add _eeg via symlink instead of renaming/moving the file.
            if !endswith(lowercase(fname), "_eeg.fif")
                base, ext = splitext(fname)
                new_fname = base * "_eeg" * ext
                new_path  = joinpath(deriv_eeg_dir, new_fname)

                if old_path != new_path
                    if is_symlink_to(new_path, old_path)
                        @info "FIF symlink already correct, skipping" new_path
                    elseif islink(new_path)
                        rm(new_path; force = true)
                        symlink(old_path, new_path)
                        @info "Updated FIF symlink target" old_path new_path
                    elseif ispath(new_path)
                        @info "Target FIF already exists as a non-symlink path, keeping as-is" new_path
                    else
                        symlink(old_path, new_path)
                        @info "Created FIF symlink" old_path new_path
                    end
                end
            end

            # 3) Create events symlink: same name but _events.tsv instead of _eeg.fif.
            events_fname = replace(new_fname, r"(?i)_eeg\.fif$" => "_events.tsv")
            events_path  = joinpath(deriv_eeg_dir, events_fname)

            if is_symlink_to(events_path, raw_events_path)
                @info "Events symlink already correct, skipping" events_path
            elseif islink(events_path)
                rm(events_path; force = true)
                symlink(raw_events_path, events_path)
                @info "Updated events symlink target" raw_events_path events_path
            elseif ispath(events_path)
                @info "Events path exists as a non-symlink file, skipping" events_path
            else
                symlink(raw_events_path, events_path)
                @info "Created events symlink" raw_events_path events_path
            end
        end
    end
end
