## Uncovering the Computational Mechanisms Underlying Many-Alternative Choice

This repository contains all code and data for:

Thomas, A., Molter, F., & Krajbich, I. (2020, March 19). Uncovering the Computational Mechanisms Underlying Many-Alternative Choice. https://doi.org/10.31234/osf.io/tk6qe

Each jupyter notebook in `src/` reproduces one of the figures of the manuscript. 


### Local installation and running the notebooks

**1. Clone and switch to this repository:**

```bash
git clone https://github.com/athms/many-item-choice.git
cd many-item-choice
```

**2. Install all dependencies** listed in [`requirements.txt`](requirements.txt). 

I recommend setting up a new Python environment (e.g., with the [miniconda installer](https://docs.conda.io/en/latest/miniconda.html)). 

You can create a new [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) using the following command:

```bash
conda create --name many-item-choice python=3.6
```

This will create a Python 3.6 environment with the name `many-item-choice`.

You can activate the new environment as follows:

```bash
conda activate many-item-choice
```

and then install all required dependencies with: 

```bash
pip3 install -r requirements.txt
```

**3. Start the Jupyter notebook server:**

```bash
jupyter notebook
```

### A few notes before computing the notebooks

This project involves many computationally intensive analyses, which can take several days to compute (depending on your machine). By default, many results are therefore read from the repository, if the respective output files exist in the `results/` directory. Thus, If you want to re-compute a specific (or all) results, delete (or rename) the respective results files / folders. 

**1. Fitting the individual models:**

To fit the individual models, run [src/fit_subject-setsize-models.py](src/fit_subject-setsize-models.py). This script fits all variants of the PSM, IAM, and GLAM for an individual subject in one set size condition. You can specify the subject and set size condition as input to the scrips:

```bash
cd src
python fit_subject-setsize-models.py <subject> <set size>
```

To fit the models for all subjects in all set size conditions, run:

```bash#
cd src
for setsize in 9 16 25 36; do
  for subject in $(seq 0 49); do
    python fit_subject-setsize-models.py $subject $setsize;
  done;
done
```

**2. Running the model recovery:**

The model recovery can be computed by the use of [src/model-recovery.py](src/model-recovery.py) script, before computing all figure supplements of Figure 5:

```bash
cd src
python model-recovery.py
```

