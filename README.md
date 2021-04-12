# Uncovering the computational mechanisms underlying many-alternative choice

This repository contains all code and data for:

Thomas, A. W., Molter, F., & Krajbich, I. (2021). Uncovering the computational mechanisms underlying many-alternative choice. Elife, 10, e57012. [doi.org/10.7554/eLife.57012](https://doi.org/10.7554/eLife.57012)

All code underlying this project is contained in [src/](src/), with one [jupyter notebook](https://jupyter.org) for each figure of the manuscript. 


## Local installation

**1. Clone and switch to this repository:**

```bash
git clone https://github.com/athms/many-item-choice.git
cd many-item-choice
```

**2. Install all dependencies:**

I recommend setting up a new Python environment (e.g., with the [miniconda installer](https://docs.conda.io/en/latest/miniconda.html)). 

To recreate our [environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) use the following command:

```bash
conda env create -f environment.yml
```

This will create a new Python 3.6 environment with the name `many-item-choice` that includes all required dependencies.

You can then activate this environment as follows:

```bash
conda activate many-item-choice
```

**3. Start the Jupyter notebook server:**

```bash
jupyter notebook
```

## The data

All data of this project are included in the [`data`](data/) repository.

There you will find two main file types:
- Aggregate data files (contained in [`data/summary_files/`](data/summary_files/)) and
- Individual subject data files (contained in [`data/subject_files/`](data/subject_files/)).  

*Note that all files are separated by choice set size (9, 16, 25, and 36).*

**Aggregate data** files contain the following information (with one row per trial): 

- `{setsize, subject, trial}` (int): indicators for the choice set size, subject, and trial
- `rt` (float): response time (in ms)
- `rt_choice_indication` (float): choice indication time (in ms; defined as time between space bar press and click on an item image; see the manuscript for details)
- `choice` (int): chosen item
- `best_chosen` (float [0, 1]): whether the subject choose the item with the highest value in that trial (1) or not (0)?
- `best_seen_chosen` (float [0, 1]): whether the subject choose the item with the highest value that they have seen in that trial (1) or not (0)?
- `longest_chosen` (float [0, 1]): whether the subject chose the item that they have looked at longest (1) or not (0) in that trial?
- `gaze_count` (float): overall number of gazes in that trial (a gaze is defined as all consecutive and uninterrupted fixations to an item)
- `returning_gaze_count` (float): overall number of returning gazes in that trial
- `seen_items_choice` (float): number of items that the subject looked at in that trial
- `item_value_{0-setsize}` (float): value of each choice set item (as indicated by the liking rating)
- `cumulative_gaze_{0-setsize}` (float): cumulative gaze of each choice set item (defined as the fraction of trial time that the subject spent looking at the item)
- `stimulus_{0-setsize}` (string): filename of respective snack food stimulus (files are stored in [`data/stimuli`](data/stimuli))
- `gaze_onset_{0-setsize}` (float): time point (in ms) at which the item was first looked at in that trial

Note that items numbers start at 0 and increase from left-to-right and top-to-bottom. The item in the top left thus has number 0 while the item in the bottom right has the highest number in the choice set,

There are two types of *individual subject data* files:

**Gaze files** contain one row for each gaze with the following columns: 

- `{setsize, subject, trial, item}` (int): indicators for the choice set size, subject, trial, and looked-at item
- `dur` (float): duration of the gaze (in ms)
- `onset` (float): onset of the gaze (in ms; relative to the trial onset)
- `stimulus` (string): filename of the snack food stimulus (files are stored in [`data/stimuli`](data/stimuli))
- `gaze_num` (float): gaze number in trial
- `is_returning` (float [0, 1]): is this a returning gaze (1) or not (0)?
- `returning_gaze_count` (float): counter for returning gazes to this item
- `is_{first,last}` (float [0, 1]): is this the first/last gaze of the trial?
- `item_value` (float): value of the item (as indicated by liking rating)
- `choice` (int): chosen item
- `is_last_to_choice` (float [0, 1]): whether chosen item was looked at last in that trial (1) or not (0)?

**Liking rating** files contain one row per rating trial with the following columns: 

- `{subject, trial}` (int): indicators for the subject and trial
- `rt` (float): response time (in ms)
- `stimulus` (string): filename of the snack food stimulus
- `rating` (float): liking rating

The folder [data/stimulus_positions](data/stimulus_positions) further contains the item positions on the experiment screen for each choice set size.

## A few notes before computing the notebooks

This project involves many computationally intensive analyses, which can take several days to compute (depending on your machine). By default, many results are therefore read from the repository, if the respective output files exist in the [results/](results/) directory. Thus, if you want to re-compute a specific result (or all of them), delete (or rename) the respective output files or folder. 

**1. Fitting the individual models:**

To fit the three model types (PSM, IAM, and GLAM) for one subject in one choice set size, run:

```bash
cd src
python fit_subject-setsize-models.py <subject> <set size>
```
This script will fit the passive- and active-gaze variants of the three models for the specified subject and choice set size.

To iteratively fit all models for all subjects in all set size conditions, run:

```bash
cd src
for setsize in 9 16 25 36; do
  for subject in $(seq 0 48); do
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

