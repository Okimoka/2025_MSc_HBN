# **Project Template:** Insights from a very large EEG+Eyetracking dataset
**Authors:** *Okan Mazlum*

**Year:** *2025*

## Project Description

The project is split into two parts:

1. Augmentation of the HBN-EEG dataset with eye-tracking data. This includes the curation (using various eye-tracking quality metrics) and synchronization of the data. This is achieved by adding custom functionality to the mne-bids-pipeline

2. Analysis of this combined dataset. Specifically it looks at the relation between eye movements and elicited neural response, and how these responses vary with age, health status, and other phenotypic factors. The analysis is done using Unfold.jl to create overlap-corrected FRPs, and later appropriate statistical tools will be used to compare these FRPs (TBD)

For more information, see the proposal in the papers folder.


## Instructions


# Instructions

- Download all 11 releases of the HBN-EEG dataset ([nemar](https://nemar.org/dataexplorer/local?search=HBN-EEG))
- Download the accompanying Eye-Tracking and Behavioral data from each release of interest ([all download pages](https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/downloads/))

For convenience, `scripts/download_dataset` contains some helper scripts along with instructions to automatically download the entire dataset and get it to the expected folder structure described in `create_merged_dataset.py`, which is a script that transforms the downloaded datasets into one combined BIDS (and therefore mne-bids-pipeline compatible) dataset (`mergedDataset`).

In order to parse the SMI Eye-Tracker format used in the HBN dataset, a modification to the mne-bids-pipeline has to be made. This repository features two variants of this modification:

- `_05b_sync_eyelink.py` which adds support for the SMI format, plus captures some important metrics about the synchronization quality and stores them in `extractedDataset/derivs/<subject>/eeg/<subject>_task-<task>-eyelink_metrics.json`. This includes some HBN-specific code, like reading the `participants.tsv` from the dataset
- `_05b_sync_eye.py` which only adds support for the SMI format and does not include any code that would be specific to only this project. It is intended as a clutter-free extension of the mne-bids-pipeline fork. When using this variant, `/path/to/mne_bids_pipeline/steps/preprocessing/__init__.py` has to be adapted to the new filename

After installing the mne-bids-pipeline fork (`pip3 install git+https://github.com/s-ccs/mne-bids-pipeline.git@temp_dev`), the `_05b_sync_eyelink.py` file has to be updated with the first of these variants (i.e. overwrite the file in `~/.local/lib/python3.12/site-packages/mne_bids_pipeline/steps/preprocessing`).

To run all processing steps up to the synchronization step, execute:

```
mne_bids_pipeline --config=config.py --steps init,preprocessing/_01_data_quality,preprocessing/_04_frequency_filter.py,preprocessing/_05b_sync_eyelink.py
```

After this step, all subjects with valid EEG and Eye-Tracking data should be populated with their respective `*-eyelink_metrics.json` files. To summarize all these files, run `python3 create_overview_from_derivs.py`, which outputs an overview file inside `extractedDataset/derivs`. A sample output is provided in `sync_metrics_overview.xlsx`. The helper scripts `histogram.py` (with a desired formula), or `score_subjects.py` can then be used to visually or numerically assess the quality of the metrics for each subject.



## Overview of Folder Structure 

```
│projectdir          <- Project's main folder. It is initialized as a Git
│                       repository with a .gitignore file.
│
├── experiment       <- Files for running the experiment
|
├── notebooks        <- Pluto, Jupyter, Weave or any other mixed media notebooks.*
|
├── papers           <- Materials related to paper, publishing
|
├── _research        <- WIP scripts, code, notes, comments,
│   |                   to-dos and anything in an alpha state.
│
├── plots            <- All exported plots go here, best in date folders.
|   |                   Note that to ensure reproducibility it is required that all plots can be
|   |                   recreated using the plotting scripts in the scripts folder.
|
│
├── scripts          <- Various scripts, e.g. simulations, plotting, analysis,
│   │                   The scripts use the `src` folder for their base code.
│
├── src              <- Source code for use in this project. Contains functions,
│                       structures and modules that are used throughout
│                       the project and in multiple scripts.
│
├── test             <- Folder containing tests for `src`.
│   └── runtests.jl  <- Main test file
│
├── README.md        <- Top-level README.
|
├── .gitignore       <- produced by Dr. Watson
│
├── (Manifest.toml)  <- Contains full list of exact package versions used currently.
|── (Project.toml)   <- Main project file, allows activation and installation. Produced by Dr. Watson
|── (project_project_template.toml)   <-  This is for a content-wise description of your experiment. 
                                          Change "project_template" in the filename to your project name. 
                        
```




# template_project - produced by Dr. Watson

TODO

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> template_project

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "template_project"
```
which auto-activate the project and enable local path handling from DrWatson.
