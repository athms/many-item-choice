## Uncovering the Computational Mechanisms Underlying Many-Alternative Choice

This repository contains all code and data for:

Thomas, A., Molter, F., & Krajbich, I. (2020, March 19). Uncovering the Computational Mechanisms Underlying Many-Alternative Choice. https://doi.org/10.31234/osf.io/tk6qe

Each jupyter notebook reproduces one of the figures of the manuscript. 


### Local installation and running the notebooks

**1. Clone and switch to this repository:**

```bash
git clone https://github.com/athms/many-item-choice.git
cd deep-learning-basics
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

**1. The "gaze_corrected" columns:**

The "gaze_corrected" columns in the summary data files (located in [data/summary_files/](data/summary_files/)) indicate the fraction of remaining trial time, after an item was seen, that the item was looked at. These values are needed for the independent evidence accumulation model (IAM). For further details, please look at the "Methods" section of our mansucript.


**2. Running fit_models.py:**

In order to compute [Figure 5](Figure-5-6_model-comparison.ipynb), you need to first run [fit_models.py](fit_models.py). This script fits all variants of the PSM, IAM, and GLAM for an individual subject in one set size condition. You can specify the subject and set size condition when running fit_models.py as follows:

```bash
python fit_models.py <subject> <set size>
```

To sequentially fit all models for all subjects in all set size conditions, run:

```bash
for setsize in 9 16 25 36; do
  for subject in $(seq 0 49); do
    python fit_models.py $subject $setsize;
  done;
done
```

**3. Running run_model_recovery.py:**

Similarly, you need for first run the [run_model_recovery.py](run_model_recovery.py) script, before computing all figure supplements of Figure 5:

```bash
python run_model_recovery.py
```

