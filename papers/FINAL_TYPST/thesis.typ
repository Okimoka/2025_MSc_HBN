// A central place where libraries are imported (or macros are defined)
// which are used within all the chapters:
#import "utils/global.typ": *
#import "utils/todo.typ": TODO
#import "utils/subfigure.typ": *
#import "utils/general-utils.typ": *

// Fill me with the Abstract
#let abstract = [#ilt[Write abstract]]

// Fill me with acknowledgments
#let acknowledgements = [
  
The authors acknowledge support by the state of Baden-Württemberg through bwHPC.

#TODO(
  color: red,
  title: "FIXME",
)[Ehlers needs acknowledgement as well, but I can't open https://wisman.izus.uni-stuttgart.de/display/SimTech/Acknowledgement]

#ilt[does simtech clusters need acknowledgement as well?]



]


// Declaration regarding own work / AI use: adapted from the guidelines of the Computer Science Department, Faculty 5, Uni-Stuttgart 
#let declaration = [
  #include "declaration.typ"
]

// if you have appendices, add them here
#let appendix = [
  = Appendices
  #include "appendix.typ"
]

// Put your abbreviations/acronyms here.
// 'key' is what you will reference in the typst code
// 'short' is the abbreviation (what will be shown in the pdf on all references except the first)
// 'long' is the full acronym expansion (what will be shown in the first reference of the document)
//
// In the text, call @eeg or @uniS to reference  the shortcode
#let abbreviations = (
  (
    key: "eeg",
    short: "EEG",
    long: "Electroencephalography",
  ),
  (
    key: "uniS",
    short: "UoS",
    long: "University of Stuttgart",
  ),
  (
    key: "et",
    short: "ET",
    long: "Eye-tracking",
  ),
  (
    key: "hbnn",
    short: "HBN",
    long: "Healthy Brain Network",
  )
  
)

#show: thesis.with(
  author: "<author>",
  title: "<title>",
  degree: "<degree>",
  faculty: "Faculty of Electrical Engineering and Computer Science",
  department: "Computational Cognitive Science",
  major: "<major>",
  supervisors: (
    (
      title: "Main Supervisor",
      name: "Jevri Hanna",
      affiliation: [Computational Cognitive Science \
        Faculty of Electrical Engineering and Computer Science, \
        Department of Computer Science
      ],
    ),
    (
      title: "Second Supervisor",
      name: "Benedikt Ehinger",
      affiliation: [Computational Cognitive Science \
        Faculty of Electrical Engineering and Computer Science, \
        Department of Computer Science
      ],
    ),
  ),
  epigraph: none,
  abstract: abstract,
  appendix: appendix,
  acknowledgements: acknowledgements,
  preface: none,
  figure-index: false,
  table-index: false,
  listing-index: false,
  abbreviations: abbreviations,
  date: datetime(year: 2025, month: 6, day: 1),
  bibliography: bibliography("refs.bib", title: "Bibliography", style: "american-psychological-association"),
  declaration: declaration
)

// Code blocks
#codly(
  languages: (
    rust: (
      name: "Rust",
      color: rgb("#CE412B"),
    ),
    // NOTE: Hacky, but 'fs' doesn't syntax highlight
    fsi: (
      name: "F#",
      color: rgb("#6a0dad"),
    ),
  ),
)

// If you wish to use lining figures rather than old-style figures, uncomment this line.
// #set text(number-type: "lining")

// import custom utilities


// Main Content starts here
= Introduction <chp:introduction>


#TODO()[Why does the thesis matter? Give context for the thesis, explain the research gap and provide a summary.]

- importance of large datasets in eeg (https://www.biorxiv.org/content/10.64898/2026.02.06.704323v1)

#ilt[something like this from presentation]
#figure()[
  #image("image-10.png", width: 60%)
] <fig:blur_plot_small>

== Background
#TODO()[Describe the background of the thesis topic, along with the research gap to be filled or the problem to be solved.]

#TODO()["What field knows" / related works - again taken from proposal, needs more detail and reworks (e.g. more related works like @Dimigen and @dimigen2 and maybe @Fusing ?)]


== Healthy Brain Network

#ilt[All texts taken from Proposal - need reworking]

The Healthy Brain Network (HBN) is an initiative by the
Child Mind Institute with the goal being #quote[to collect data from
and provide diagnostic consultations to thousands of children
and adolescents] @hbnabout. As part of this effort, one subset of their
collected and publicly released data contains the @eeg data (128
channels at 500 Hz), @et (Eye position and pupil dilation at
120 Hz) and behavioral data of more than 4000 subjects (ages
5-21), making it one of the largest research-grade, coherent
EEG datasets available @hbn @fair. 

#ilt[quick overview table over the data]
- mention iView-X Red-m by SensoMotoric Instruments
- 935 subjects did not have any ET data (only EEG)

The authors have highlighted the importance and effectiveness of analyzing such large-scale
EEG datasets for robust scientific results, naming this as part
of their motivation @hbn. Numerous studies have already been
conducted using the HBN EEG data @Dubois.2025 @Lasnick.2023 @Jia.2019 @Jiang.2024, but only few make use of the available ET data @eyes1 @eyes2, one
reason possibly being the lack of BIDS integration.


#figure(caption: [Diagnoses of the subjects in the HBN dataset])[
  #image("subjects_diagnoses.png", width: 100%)
] <fig:subjects_diagnoses>

The HBN dataset itself encompasses six different paradigms
that each of the subjects were tested on (with exceptions).
Three of the tasks were active (i.e. required participant
interaction) and three were passive. Since these tasks are the
foundation of our analyses, each paradigm is described in
closer detail here:


#ilt[
  mention 5-point calibration that was performed at the start of each paradigm
]




//#figure(caption: [Exemplar runs of paradigms that were performed for the Healthy Brain Network (HBN) dataset. Images are modified
//from @fair])[
//  #image("MNE_paradigms.png", width: 100%)
//] <fig:MNE_paradigms>

=== Contrast Change

#subfigure(
  figure(
    image("image-4.png"),
    caption: [Exemplar contrast change detection sequence with participants response (s)],
  ),
  <fig:contrastchange_left>,
  figure(
    image("placeholder.png"),
    caption: [TODO],
  ),
  <fig:contrastchange_right>,
  columns: (1fr, 1fr),
  caption: [Contrast change task visualized],
  label: <fig:contrastchange>,
)




Two circular overlapping gratings (black and white striped patterns) are
shown on a screen (@fig:contrastchange_left). One is flickering at 20 Hz and is tilted 45°
to the left, the other one is flickering at 25 Hz and tilted 45°
to the right. The subjects were asked to maintain fixation on
the center of the gratings. At some point, the contrasts of the
two gratings gradually (within 1.6s) go from each being 50%
to being of 0% and 100% contrast respectively, after which
they go back to normal within 0.8s. The participants had to
monitor which of the two gratings is beginning to fade and
press a corresponding button as soon as they can identify the
correct side. Feedback is given by flashing a sad or happy face.
===  Sequence Learning

#subfigure(
  figure(
    image("image-7.png"),
    caption: [Timeline of one sequence learning block (s)],
  ),
  <fig:sequencelearning_left>,
  figure(
    image("placeholder.png"),
    caption: [TODO],
  ),
  <fig:sequencelearning_right>,
  columns: (1fr, 1fr),
  caption: [Sequence learning task visualized],
  label: <fig:sequencelearning>,
)

A random visual stimulus
representing a number between 1-8 is flashed on
a screen (@fig:sequencelearning_left) and immediately fades out within 0.2s. After a 1.3s
pause this process repeats, resulting in a 10-item sequence
(8 items for ages 8 and below). Again, the subjects were
asked to maintain fixation on the center of the visual stimuli.
The sequence is then recalled (with indefinite time limit and
without feedback) using a computer mouse. This "Observe
→ Recall" cycle is repeated five times with the exact same
sequence.
===  Symbol Search

#subfigure(
  figure(
    image("image-8.png"),
    caption: [Exemplar 3 questions of the sequence learning task. Participants were instructed to click on Yes if any of the target shapes were present in the search group],
  ),
  <fig:symbolsearch_left>,
  figure(
    image("gaze_history.png"),
    caption: [Gaze history of an exemplar subject solving the first page of questions. Each dot is a saccade],
  ),
  <fig:symbolsearch_right>,
  columns: (1fr, 1fr),
  caption: [Symbol search task visualized],
  label: <fig:symbolsearch>,
)
Participants are shown 15 rows
of geometric symbols (@fig:symbolsearch_left). Each row contains two
target symbols and five search symbols. For each row, the
participant must decide if either of the target symbols is
present in the search group and indicate their choice by
clicking a "yes" or "no" checkbox with a computer mouse.
When done, the participants clicked a "next page" button to
view the next group of rows. The task has a 2 minute time
limit to complete as many rows as possible. In a brief training
session participants received feedback on their performance,
but not in the main task.
=== Resting-State 
Participants focus on a fixation cross
shown on a screen. They alternate between keeping their eyes
open for 20 seconds and keeping them closed for 40 seconds,
following auditory instructions. This cycle of eyes-open and
eyes-closed is repeated five times.
=== Surround Suppression 
#subfigure(
  figure(
    image("image-5.png"),
    caption: [An exemplar surround suppression sequence (s)],
  ),
  <fig:surroundsuppression_left>,
  figure(
    image("placeholder.png"),
    caption: [TODO],
  ),
  <fig:surroundsuppression_right>,
  columns: (1fr, 1fr),
  caption: [Surround suppression task visualized],
  label: <fig:surroundsuppression>,
)
Participants focus on
a fixation cross shown on a screen (@fig:surroundsuppression_left). For each trial, the
participant is shown a full-screen "surround" grating that
has four circular "foreground" gratings overlaid within. Each trial randomizes the contrast of the
foreground (0%, 30%, 60%, or 100%), the surround (0% or
100%), and the surround orientation (vertical or horizontal).
Each trial takes 2.4 seconds and the experiment took place in
two blocks of 64 trials each.
=== Naturalistic Viewing
Participants are asked to view
and enjoy 4 video clips: "E-How video: How to Improve at
Simple Arithmetic: Lessons in Math" (1:40), "MIT K-12:
‘Fun with Fractals’" (4:40), Diary of a Wimpy Kid Trailer
(2:00) and "Despicable Me" (2:50).

#figure(
  table(
    columns: 5,
    table.header(
      [Paradigm name],
      [Active?],
      [Free viewing?],
      [\# of recordings],
      [Mean duration],
    ),

    [Resting State], [No], [No], [2880], [06:36],
    [Surround Suppression], [No], [No], [4150], [06:52],
    [Contrast Change], [Yes], [No], [1797 x 3], [5:55 x 3],
    [Sequence Learning], [Yes], [No], [1890], [05:25],
    [Symbol Search], [Yes], [Yes], [1567], [03:42],
    [Naturalistic Viewing], [No], [Yes], [2699 x 4], [3:03],
  ),
  caption: [Paradigm overview],
)





== Brain Imaging Data Structure
The Brain Imaging Data Structure (BIDS) is a specification
for formatting and describing neuroimaging experiments in
a unified and standardized way, mitigating the necessity of
tailoring tools to a specific format. Tools such as Unfold.jl as
well as the MNE-BIDS-Pipeline already fully support BIDS.

#TODO()[The standard is constantly evolving, with an integration of an ET data specification #strike[currently in the works @bidsET] having happened during the course of this thesis @BIDS_ET] 


BIDS also supports HED (Hierarchical Event Descriptors)
annotations which allow detailed, machine-readable descrip-
tions of the experimental events, making it easier to perform
automated analyses. Part of the HBN data such as the imaging
data (MRI) is already in BIDS format, but most of the dataset
is in currently in the process of being converted over.


== HBN-EEG: The FAIR implementation of the Healthy Brain Network electroencephalography dataset
HBN-EEG @fair is an effort to convert the HBN dataset to BIDS,
which the authors describe as #quote[starting point in providing a
large, transparent dataset in a form that will assist researchers
in easily identifying the information they need to pursue their
research [...]]. The EEG data from all 11 Releases of the
HBN dataset has already been converted. Each subject run was
labeled with either ”Available”, ”Caution” or ”Unavailable”,
corresponding to usable runs, runs that need cleaning up (failed
at least one consistency test), and runs that failed to convert. A
total of 525 subjects did not have a single usable run, resulting
in ”available” data from 3613 subjects. Some further results
include:
- Some behavioral data has already been integrated and synchronized as events.
- Events have been annotated using HED (Hierarchical Event Descriptors).
- Phenotypic data has been summarized for easier analysis (P-factor, Attention, Internalizing, Externalizing).
- An open-source automated electrode localization toolbox is being developed for digitizing electrode positions of the 2270 HBN participants who have had their electrode locations scanned

#strike[Finally, while the full integration of the Eye-Tracking data
was currently bottlenecked by the unfinished Eyetracking-BIDS
specifications, the tools to perform the conversion to BIDS are
currently in beta @eegbids and will be used in this thesis.]

== Related Works
=== Regression-based analysis of combined EEG and eye- tracking data: Theory and applications

The authors demonstrate a state-of-the-art methodology in
overcoming the difficulties posed by the analysis of FRPs (especially in free-viewing conditions), which they have broken
down to
+ the synchronization of data streams
+ the removal of ocular artifacts
+ the condition-specific temporal overlap between the brain responses evoked by consecutive fixations
+ the fact that numerous low-level stimulus and saccade properties also influence the postsaccadic neural responses
The task is further complicated by the fact that:
- target stimuli are often fixated on for longer timespans than non-targets, creating a spurious effect between conditions
- a similar bias occurs when the conditions have different average saccade amplitudes (e.g. target conditions consistently require a bigger shift in gaze)
- many paradigms flash in a visual stimulus at the beginning, causing a long lasting potential that distorts the following signals (stimulus-onset ERP)
- factors like the saccade amplitude have a nonlinear effect on the FRP
- for active tasks, movements like button presses will further distort the signal
- involuntary miniature saccades are produced even when subjects are asked to fixate on one point

The authors argue that a framework combining linear
deconvolution with nonlinear regression can effectively
address and mitigate most of these issues. By modeling the
continuous EEG signal as a sum of overlapping individual
brain responses, it is possible to account for confounding
variables by making sure they are part of the regression
formulae. For example:


`ERP ∼ 1` adresses the stimulus onsets

`FRP ∼ 1 + spl(fixation_position_x, 5) + spl(fixation_position_y, 5) + spl(sacc_amplitude, 5) + circspl(sacc_angle, 5, 0, 360)` addresses
(among others) the nonlinear effect of saccade amplitude and the effect of saccade direction.

The effectiveness is demonstrated in three different experiments: Face perception, scene viewing, and reading. Synchronization was done using EYE-EEG @eyeeeg and for the
deconvolution and regression, the Unfold toolbox @Unfold was
used.
Each of the three experiments highlights a different use case
or strength of the approach in combatting the issues mentioned
above:
- Face perception: Subjects were instructed to classify the emotional expression of a face use a button press. Despite being asked to fixate on a single point, 99% of trials had subjects perform miniature saccades (microsaccades or small exploratory saccades). The deconvolution model was able to isolate the (brain and muscle) potentials caused by these miniature saccades, and in turn was able to lead to a cleaner ERP
- Scene viewing: Subjects searched for a small, but constantly growing, gray dot inside various grayscale photographs. When they found it, they would press a button. This free-viewing experiment contained multiple nonlinear confounding variables, as well as stimulus onsets and button presses. Still, the deconvolution model managed to successfully separate these effects.
- Natural reading: Subjects read two sentences with one containing a ”target word”. In some trials, this target word was static. In other trials, the target word was instead a random but visually similar string, and was only replaced with the correct target word during a saccade just before reading the target word. This manipulation causes shorter fixations on words with valid preview. The challenge is to determine whether there was an actual neural difference in the two conditions (valid or invalid preview), or whether differences in the signal are just caused by difference in fixation length. Using their framework, it could be confirmed that there is an actual neural difference between the two conditions. #ilt[mention how this is relevant to the symbol search task, where there should be a spurious effect between conditions, as the target symbols will be fixated on for longer timespans (needs to be measured)]

In their analyses of each paradigm, the authors employ a
two-stage statistical approach:
+ Individual Level: The deconvolution regression model is computed separately for each participant using Unfold. This results in the regression coefficients (betas, as noted by the β vector in the regression model), which can be understood as the separation of the various confounding factors and the cleaned FRP waveforms. These waveforms can now be treated like a regular subject-level ERP.
+ The betas are compared in within-subjects analysis to test for effects across all participants. The specific test that is used to identify significant effects in the EEG data is threshold-free cluster-enhancement (TFCE), and it has the advantage of controlling for multiple comparisons across all time points and electrode channels.





#image("image.png")

== Research questions
#TODO()[You may write about your research questions here.]

== Summary
#TODO()[Sum up the approach and results.]
#image("image-1.png")
//
// NOTE:
// It's important to have explicit pagebreaks between each chapter,
// otherwise header stylings from the template might break
#pagebreak()
= Dataset preparation

#ilt[Properly write out]

In addition to the usual preprocessing performed on EEG datasets, we first had to prepare a cleaned up and unified version of the data provided in the HBN and HBN-EEG datasets (elaborate: lack of synced ET data, split into 11 releases which makes reading with mne-bids-pipeline / UnfoldBIDS very clumsy, various minor omissions, ...)

#ilt[adjust image (make svg + steps more in line to thesis structure) and caption]
#figure(caption: [The main tasks for this thesis])[
  #image("/thesis_task_steps.png", width: 100%)
] <fig:blur_plot_small>



== Unification of HBN and HBN-EEG datasets

- The HBN-EEG dataset was retrieved from NeMAR @nemar-hbn-eeg and consists of 11 `.zip` archives totalling 1.7 TiB, each containing curated EEG data in BIDS format.

- The original HBN dataset is hosted on the public fcp-indi S3 bucket @bucket. The folder `data/Archives/HBN/EEG/` contains 4576 `.tar.gz` subject archives totalling 5.6 TiB. Each subject archive holds at most 3 Folders, Behavioral, EEG and Eyetracking #ilt[double check logs whether subjects exist where not all 3 are present].
  - `Behavioral`: Contains phenotypic data about the subject in at most two different formats, `.csv` and `.mat`.
  - `EEG`: Contains preprocessed #footnote[The exact preprocessing steps that were applied are unknown and the preprocessed data is not meant to be used
https://www.nitrc.org/forum/forum.php?forum_id=10003&thread_id=15454] and raw EEG data. Since the HBN-EEG dataset will be used over the original HBN dataset for EEG data, this folder is of no further interest.

  - `Eyetracking`: Contains at most 2 subfolders out of: `idf`, `txt`, and `tsv`. Each folder represents the same eyetracking data in a different format.
    - `tsv` is not suitable for the synchronization method described in @app:syncs0, as it lacks event markers
    - `idf` is the proprietary format by SMI Vision that would have to be read in using a tool like @readsmi. Fortunately, all subjects containing a `idf` folder, also contain a `txt` folder.
    - `txt` contains eyetracking data in a format more closely described in @app:syncs

Based on the release documentation @hbn-releases, updates between releases mostly concern data availability and dataset curation (e.g., additions of participants, corrections to phenotypic tables or metadata). In particular, there are no changes that would require release-specific treatment of the EEG data.
For this reason, EEG data from the full HBN-EEG collection were merged together to form a single
integrated BIDS dataset. This enables a single, consistent preprocessing and analyis pipeline
across all subjects. This merged dataset was then enriched with the associated eye-tracking and
phenotypic data for each subject from the original HBN archives. #strike[The integration of this type
of data is not yet standardized in BIDS], and this step will cause BIDS validators @bids-validator to consider
the dataset as invalid. The tools used in this thesis (mne-bids-pipeline and Unfold.jl) remained
compatible with this developmental format, simply ignoring the extra files that are not of relevance.
The following sections cover how this merged dataset was created.

=== Merging of all Releases

#ilt[write out]



- Download is 1.7 TiB + 11 TB. After stripping out unnecessary files, the dataset is 2.2 TB in size.

- Custom helper scripts were used to download and unpack all required source archives in a restartable manner. This was done to improve the reproducibility of the study compared to manual downloading. Downloads were performed with parallel workers and to ensure integrity of the files, incomplete transfers were written to a staging directory and only moved to the final location after the expected file size was reached. This approach was taken both for the retrieval from the fcp-indi S3 bucket and for the web download from NeMAR. After download, all archives were extracted using dedicated unpacking scripts making use of the same paradigms to avoid partial extractions and make use of parallelization. For more information, see @app:repro-tips

#ilt[check exact code - some more info in midterm presentation script + slides]
Finally, further scripts make use of the downloaded files to deterministically create the final
merged BIDS dataset. For files that are required as-is in the new merged dataset, e.g. EEG and
eye-tracking recordings, the script creates symlinks to the downloaded files and renames them to
be BIDS conform. Other files like the participants.tsv or other metadata files for the merged
dataset have to be generated by merging metadata files from the separate HBN-EEG releases (i.e.
concatenating all participants.tsv or some summary tables together into one file). To support
traceability, some files were also augmented with their source release numbers. For files that were
contained in all releases with only minor differences irrelevant to the merged result, the file from
the first release was chosen.

Because metadata conventions and file naming practices vary slightly across releases, some harmonization steps were applied during dataset construction. These include:

#ilt[Go into slightly more detail]
(give examples for the filename harmonization and edge case subjects, list consistency/validity checks)

- Standardization of BIDS file names (subject identifiers, task labels, run indices, and separators) 
- Completion and canonicalization of `∗_channels.tsv` contents by adding required columns and enforcing consistent channel types and units
- Normalization of tabular files by harmonizing delimiters, column names, and column order
- Targeted handling of a small number of subjects (< 20) with irregularities beyond these standard cases (e.g. by hardcoded renaming, conversion, or exclusion).
Throughout the merging process, the scripts validated expected directory structure and file
contents, logging and/or terminating on any unexpected finds
=== Synchronization of EEG and ET  <app:syncs0>
#ilt[Write out]
- Synchronization works by lining up the timestamps for events in the ET recording and EEG recordings
#ilt[Does number of events actually notably positively impact sync quality?]



==== Synchronization using EYE-EEG
- Is the tool that was used in @Dimigen
- Is natively compatible with the SMI format
- Has integrated synchronization quality checks and metrics (`checksync.m`)
- Automatic rejection of out-of-range gaze values
- Simple automatic EEG quality check using peak-to-peak voltage differences in shifting window
- Conversion to BIDS using EEG-BIDS

=== Synchronization using mne-bids-eyetracking-pipeline <app:syncs>

#ilt[Intro to mne-bids-pipeline + the fork]

- (The derived `.fif` files contain eye-tracking samples and channels alongside the usual EEG data)

==== Parsing SMI `.txt` files

The used mne-bids-pipeline fork exclusively supported eye-tracking files in eyelink format (`.asc`,
`.edf`), and had to be extended with the functionality to parse the `.txt` files.

SMI Vision no longer exists, and there does not seem to be existing tooling for SMI -> Raw conversion (eyelink has much better support, e.g. https://github.com/bids-standard/eye2bids)

#ilt[Description of the format]:

An eye-tracking recording has two files associated with it.

- *Samples file* `sub-<id>_task-symbolSearch_et.txt`: #ilt[briefly describe structure] (essentially, the file starts with some headers containing a few useful pieces of information, followed by a tsv (with slight quirks). this table lists all ET samples as well as all user events with their respective timestamps)
- *Events file* `sub-<id>_task-symbolSearch_et_Events.txt`: #ilt[briefly describe structure]: (again, some headers followed by a quasi-tsv. this time the tsv contains all user events (these were confirmed to be fully identical to the user events in the samples file) and saccade, blink, and fixation events. these events include fixation duration, position, etc...). Mention how the events were determined (software named in @hbn2)

A total of 120 subjects are missing Samples files for all of their task recordings. Since this data is necessary for judging synchronization quality, these subjects were skipped for all analyses. The reverse case (Samples file with missing Events file) never occurs.

#ilt[give quick summary of read_raw_iview function]


== Curation of ET recordings

Throughout the parsing and synchronization process, metrics were collected to assess the quality
of the input EEG and Eye-tracking files, as well as the resulting synchronized file. These
metrics include meta properties, basic validity checks
 and previously built-in synchronization quality feedback. (For a full list of metrics, see @fig:all_metrics, for now we will only focus on the metrics that actually ended up being used for curation)

#figure(caption: [Default metrics of the mne-bids-eyetracking-pipeline])[
  #image("metrics_default-1.png", width: 100%)
] <fig:mne_metrics_default>

(quickly explain the different metrics captured here TODO)

(Argue that we need automatic evaluation due to large nature of the dataset. Hence the attempt to numerically judge the quality of the xcorr plot TODO)

=== Cross correlation plot

The cross-correlation is computed between the horizontal EOG (HEOG) signal and the eye-
tracker's horizontal gaze position (“L POR X [px]” and “R POR X [px]”, averaged if both are present). If both streams are correctly time-aligned, EOG potential and gaze position movement should co-occur and produce a strong correlation at (or very near) zero lag.





#subfigure(
  figure(
    image("metrics_pyramid.png", width: 80%),
    caption: [Cross-correlation graphs with "pyramid" metric],
  ),
  <fig:philosophers>,

  figure(
    image("metrics_gauss.png", width: 80%),
    caption: [Cross-correlation graphs with gauss fit metric],
  ),
  <fig:dining_philosophers>,

  columns: (1fr,),
  caption: [Alternative approaches to rating cross-correlation curves],
  label: <fig:with_subfigures>,
)

Once finished, the script writes all metrics as a as `∗_metrics.json` into the derivatives folder next
to the standard mne-bids-pipeline output. Additionally, an `∗_xcorr-artifact.npz` file stores part
of the cross-correlation plot. A helper script combines all `.json` files to generate an overview
over every recording as `.csv`

#ilt[document how this was evaluated]








== Preprocessing of the unified dataset
#ilt[Write out]
- Go over parameters of config.py
  - Briefly mention initial between rest_epochs_duration and epochs_tmax
- Issues with ICA decomposition
  - add merged mock tasks freeView + allTasks (use symlinks) for more ICA training 
#TODO(
  color: red,
  title: "Ask Bene",
)[What was the paper that gave an estimate of how much ICA training data is required?]
- Why was each parameter chosen as it was? e.g.
  - ica_reject:
    + so that when concatenating tasks, the same epochs get rejected (relevant for the mock tasks) unlike with autoreject
    + autoreject was too sensitive (probably not tuned to freeview tasks, where large ET artefacts are to be expected)
  - ica_use_ecg/eog_detection turned off because icalabel seemed to be more accurate
  - ...




#pagebreak()
= Methods
#TODO()[Describe the method(s) used.]
#image("image-2.png")

#ilt[Rewrite]

== Unfold

The Unfold toolbox is a family of tools relating to EEG
processing and analysis. It has been used successfully in
analyzing fixation-related potentials (FRPs) from combined
eye-tracking/EEG experiments similar to e.g. the naturalistic
viewing paradigm of the HBN dataset @Dimigen. This thesis
will use the Unfold toolbox, in particular its (non)linear
deconvolution capabilities for its main analysis.

#ilt[go into more detail on how exactly unfold works]

One precondition for deconvolution to work is sufficient
variation in fixation duration, saccade amplitude and event
sequences (saccade rate)

#ilt[distribution for subjects that were eventually analyzed]



== Execution details
#TODO()[If applicable, explain how the method was followed.]

#ilt[Properly integrate into corret position / text]

- symbol search and naturalistic viewing will be the main focus of the analysis, as all other tasks had subjects fixate on a single point, and the Eye Tracker used for the HBN dataset is unsuitable for miniature saccade analysis. This is in large due to its low sampling rate of 120 Hz and gaze position accuracy of 0.5°, which are both of insufficient resolution for these kinds of saccades (by definition of miniature saccades).
- firbasis window was chosen based on @Dimigen's (-0.4 to 0.8), but extended to (-0.5, 1) due to the observed phenomena (dip before 0, lasting voltage after peak)
- Plotting was done either using matplotlib (version X) in python 3.12 or (Unfold)Makie (versions X) in julia version X


== Dataset size hurdles

- mne-bids-pipeline took multiple minutes to start up - document changes that had to be made to avoid large scans


= Results
#TODO()[Present, describe and summarize the results obtained.]

#TODO()[erpimage plot like @Dimigen, with single trials sorted by the duration of the current fixation. Check whether the brain response from the subsequent fixation actually starts earlier for trials with shorter fixations]

#TODO()[erpimage residuals plot (EEG activity not covered by the model) like @Dimigen, with single trials sorted by the duration of the current fixation. Argument that this can highlight
whether saccade-related activity from neighboring fixations is properly filtered]

#TODO()[Un-corrected, raw FRP plot (for baseline subjects)]

#TODO()[Plot with unusual patterns circled
- visually evoked lambda response of saccades does not match the 110 ms after saccade onset given in @Dimigen
- saccade amplitude not only doesn't have a nonlinear effect on the lambda response, the peak seems to be almost the same height regardless of amplitude. 
- dip depth is dependent on amplitude, likely because larger amplitudes create stronger artifacts
- higher amplitudes have "lingering" voltage after the lambda response peak
]



#pagebreak()
= Discussion

#image("image-3.png")

== Interpretation
#TODO()[Write about the interpretation of the results and explain how the identified research gap was filled.]

== Limitations
#TODO()[What limitations does the interpretation have? How can this be fixed?]

== Outlook / Future Scope
#TODO()[What difference was made by the work done on the thesis, and how can it be expanded on in the future?]

#ilt[Properly formulate]
  - include local luminance and contrast into regression formula
  - In the HBN-EEG paper, the authors suggest some further improvements that could be made to the HBN-BIDS dataset other than the integration of ET data, for example the annotation of movies with detailed event sequences.
  - After this, it would be easy to add a categorical variable is_face() to the regression model
 - Different ICA algorithm (e.g. OPTICAT)
 - More modern Eye event detection algorithm
 #TODO(
  color: red,
  title: "Ask Bene",
)[Use modern experimental algorithms that would allow the equivalent of epoch rejection but on continuous data (which algorithm was referred to?)]
- Use task performance (e.g. fast detection of contrasts in contrast change paradigm) as a comparison variable for subjects
- Papers that have used datasets with the same paradigms as the HBN dataset, could be attempted to be challenged/verified by using a much larger sample size.
- As an ambitious goal, it could be compared how well AI techniques perform in the same analyses vs. the established methodology



#pagebreak()
= Summary
#TODO()[Recap what was discussed.]

//= Other chapters
//The further chapters give detailed information on the template. They have been included from a separate typst file and can be omitted by commenting out or deleting the `#include` line following this chapter in the source code. 


//#include "template/demo/main.typ"
#include "declaration.typ"