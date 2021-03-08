# Uncovering the computational mechanisms underlying many-alternative choice

This repository contains all code and data for:

Thomas, A., Molter, F., & Krajbich, I. (2020, March 19). Uncovering the Computational Mechanisms Underlying Many-Alternative Choice. https://doi.org/10.31234/osf.io/tk6qe

All code underlying this project is contained in [src/](src/); Each jupyter notebook in reproduces one of the figures of the manuscript. 


## Local installation

**1. Clone and switch to this repository:**

```bash
git clone https://github.com/athms/many-item-choice.git
cd many-item-choice
```

**2. Install all dependencies:**

I recommend setting up a new Python environment (e.g., with the [miniconda installer](https://docs.conda.io/en/latest/miniconda.html)). 

To recreate our [Anaconda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) use the following command:

```bash
conda env create -f environment.yml
```

This will create a Python 3.6 environment (including all dependencies) with the name `many-item-choice`.

You can activate the new environment as follows:

```bash
conda activate many-item-choice
```

**3. Start the Jupyter notebook server:**

```bash
jupyter notebook
```

## The data

All data of this project is included in the [`data`](data/) repository.

There you will find two types of main files: aggregate data (contained in [`data/summary_files/`](data/summary_files/) and individual subject data (contained in [`data/subject_files/`](data/subject_files/). 

All files are separated by choice set sizes (9, 16, 25, and 36). 

Aggregate data files contain the following information (each row indicates on experiment trial): 

- `{setsize, subject, trial}` (int): indicators for the choice set size, subject, and trial
- `rt` (float): response time (in ms)
- `rt_choice_indication` (float): choice indication time (in ms; defined as time between space bar press and click on an item image)
- `choice` (int): chosen item (items numbers start at 0 and increase from left to right and top to bottom)
- `best_chosen` (float [0, 1]): whether the subject choose the item with the highest liking value in that trial (1) or not (0)?
- `best_seen_chosen` (float [0, 1]): whether the subject choose the item with the highest liking value in that trial from the set of items that they have looked at (1) or not (0)?
- `longest_chosen` (float [0, 1]): whether the subject chose the item they have looked at longest (1) or not (0)?
- `gaze_count` (float): number of gazes in that trial (a gaze is defined as all uninterrupted fixations to an item)
- `returning_gaze_count` (float): number of returning gazes in that trial
- `seen_items_choice` (float): number of items that subject looked at in that trial
- `item_value_{0-setsize}` (float): liking rating value of each item in that trial
- `cumulative_gaze_{0-setsize}` (float): cumulative gaze of each item in that trial
- `stimulus_{0-setsize}` (string): filename of snack food stimulus (files are stored in [`data/stimuli`](data/stimuli)) of each item
- `gaze_onset_{0-setsize}` (float): time point (in ms) after trial onset at which item was first looked at

There are two types of individual subject data files:

Gaze files contain the following information (each row indicates on gaze): 

- `{setsize, subject, trial, item}` (int): indicators for the choice set size, subject, trial, and looked-at item
- `dur` (float): duration of gaze (in ms)
- `onset` (float): onset of gaze after trial onset (in ms)
- `stimulus` (string): filename of snack food stimulus
- `gaze_num` (float): number of gaze in trial
- `is_returning` (float [0, 1]): is this a returning gaze (1) or not (0)?
- `returning_gaze_count` (float): counter of returning gazes for this item
- `is_last` (float [0, 1]): is this the last gaze of the trial?
- `is_first` (float [0, 1]): is this the first gaze of the trial?
- `item_value` (float): liking rating of looked-at item
- `choice` (int): chosen item in that trial
- `is_last_to_choice` (float [0, 1]): whether chosen item was looked at last in that trial (1) or not (0)?

Liking rating files contain the following information (each row indicates one experiment trial): 

- `{subject, trial}` (int): indicators for the subject and liking rating trial
- `rt` (float): response time (in ms)
- `stimulus` (string): filename of snack food stimulus
- `rating` (float): liking rating


### A few notes before computing the notebooks

This project involves many computationally intensive analyses, which can take several days to compute (depending on your machine). By default, many results are therefore read from the repository, if the respective output files exist in the [results/](results/) directory. Thus, if you want to re-compute a specific result (or all of them), delete (or rename) the respective output files or folder. 

**1. Fitting the individual models:**

To fit the individual models, run [src/fit_subject-setsize-models.py](src/fit_subject-setsize-models.py). This script fits all variants of the PSM, IAM, and GLAM for an individual subject in one set size condition. You can specify the subject and set size condition as input to the scrips:

```bash
cd src
python fit_subject-setsize-models.py <subject> <set size>
```

To fit the models for all subjects in all set size conditions, run:

```bash
cd src
for setsize in 9 16 25 36; do
  for subject in $(seq 0 49); do
    python fit_subject-setsize-models.py $subject $setsize;
  done;
done
```

**2. Running the model recovery:**

The model recovery can be computed by the use of [src/model-recovery.py](src/model-recovery.py):

```bash
cd src
python model-recovery.py
```

Note that the model recovery is based on the individual model fits for the 9-item choice sets and can thus only be computed if these fits exist in [`results/posterior_traces`](results/posterior_traces/)!

