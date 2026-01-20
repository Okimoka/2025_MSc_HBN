import mne

#0 = symbolsearch
#1 = freeView (all merged)
EXPERIMENT = 0

bids_root = "mergedDataset"
deriv_root = "mergedDataset/derivs"
subjects_dir = None

#subjects = ["NDARAB678VYW"]
#subjects = ["NDARJA830BYV","NDARHP841RMR","NDARVD609JNZ","NDARAY461TZZ","NDARKT714TXR","NDARXU902BRT","NDARMF116AFR","NDARWB782FLR","NDARGN483WFH","NDARNH200DA6","NDARKX665ZD3","NDARVT488RXJ","NDARFY623ZTE","NDARRB942UWU","NDARRK135YAZ","NDARHZ413DZL","NDARCT889DMB"]
subjects = ["NDARJA830BYV"]

ch_types = ["eeg"]
interactive = False
sessions = []
task = ["symbolSearch","freeView"][EXPERIMENT]
#task = "symbolSearch"
#task_et = "WISC_ProcSpeed"

task_is_rest = True
runs = ["1", "all"][EXPERIMENT]
et_has_run = False
et_has_task = True

epochs_tmin = 0 #forced for rest 
epochs_tmax = 1
rest_epochs_duration = 1 #forced for rest, make small so not too much gets rejected
rest_epochs_overlap = 0 #forced to be set
baseline = None # for later Unfold analysis

#raw_resample_sfreq: float | None = 250
# plenty of papers use these values for HBN EEG
# dimigen+ehinger paper uses 0.1 Hz for their FRPs
l_freq: float | None = 0.1 
h_freq: float | None = 100

# determined by icalabel
ica_h_freq: float | None = 100
ica_l_freq = 1
eeg_reference = "average"

# data was recorded in the US
notch_freq = 60

on_error = "continue"

######### Remove these when doing Unfold analysis! ############

# positive / negative feedback
#conditions = ["HAPPY", "SAD"]
########conditions = ["# Message: 12XX", "# Message: 13XX"]

########epochs_tmin: float = -0.5
########epochs_tmax: float = 2.6 # since feedback is so infrequent, long ########epochs are okay
########
########baseline: tuple[float | None, float | None] | None = (-0.2, 0)
###############################################################


spatial_filter = "ica"
# ica_n_components = 96 ?
# icalabel forces picard-extended_infomax or extended_infomax
ica_algorithm = "picard-extended_infomax"
#ica_use_ecg_detection: bool = True
#ica_use_eog_detection: bool = True
ica_use_icalabel = True
#ica_reject: dict[str, float] | Literal["autoreject_local"] | None = "autoreject_local"

#ica_reject = "autoreject_local"
ica_reject = dict(eeg=200e-6)

#reject = "autoreject_local"

sync_eyelink = [True, False][EXPERIMENT]
#sync_eventtype_regex = "\\d-trigger=10 Image moves"
#sync_eventtype_regex_et = "trigger=10 Image moves"

#Contrast detection
#sync_eventtype_regex     = r"contrastTrial_start"
#sync_eventtype_regex_et  = r"# Message: 15"

# does it make sense to sync over full freeViewing? Probably not
sync_eventtype_regex     = r"(?:trialResponse|newPage)"
sync_eventtype_regex_et  = r"# Message: (?:14|20)"

#eog_channels = ["HEOGL", "HEOGR", "VEOGL", "VEOGU"]


#eeg_bipolar_channels = {"HEOG": ("HEOGL", "HEOGR"), "VEOG": ("VEOGL", "VEOGU")}
#eog_channels = ["HEOG", "VEOG"]
#sync_heog_ch = ("HEOG")



#eeg_bipolar_channels = {"HEOG": ("E40", "E109"),
#                            "VEOG": ("E21",  "E127")} #left eye
##eeg_bipolar_channels = {
##    #"HEOG": ("E127", "E126"),
##    "HEOG": ("E126", "E127"),
##    "VEOG": ("E22", "E127"), #left eye
##}


eeg_bipolar_channels = {
    "HEOG": ("E8", "E25"),   # left vs right outer canthus
    "VEOG": ("E21", "E17"),  # nasion vs left inner canthus

}


eog_channels = ["HEOG", "VEOG"]

sync_heog_ch = "HEOG"

sync_et_ch = ("L POR X [px]", "R POR X [px]")


#sync_et_ch = "xpos_right"
sync_plot_samps = 3000

decode: bool = False
run_source_estimation = False

montage = mne.channels.make_standard_montage("GSN-HydroCel-128")

eeg_template_montage = montage
drop_channels = ["Cz"]

n_jobs = 2