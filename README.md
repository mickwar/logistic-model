# Application Processing

## Data

Training data consists of 125000 rows with 122 columns (about 67 Mb).
The objective is to make predictions for the binary variable called `TARGET`.

The test data consists of 48744 rows with 121 columns (`TARGET` is omitted).

The input data consists of 120 columns (no `TARGET` and no ID column) of which 104 are numeric variables and 16 are categorical.

Of the 125000 rows in the training data, only 3461 contain zero missing values.
Most rows have between 5 and 49 missing values.
The largest number of missing values in a single row is 61.

## Procedure

Cross-validation on the training set.
Split the training set into `k=10` subsets.
For each subset, fit a model to the nine other subsets and make predictions on the held-out subset.
The predictions can be compared with the actual `TARGET` so care can be taken to not overfit the model.

## Model

## Results

## Using the virtual environment

Initialize the Python virtual environment in the command line with

```bash
virtualenv env --python=python3.6	# Creates the environment with a Python 3.6 binary
source env/bin/activate				# Activates the environment
pip install -r requirements.txt		# Installs the packages found in requirements.txt
```
This will install packages such as Keras, Numpy, and Pandas into a self-contained environment, avoiding any conflicts with globally installed packages.
The environment can also be safely deleted by removing the `env` directory.

When doing any work with the code, be sure to run
```bash
source env/bin/activate
```
so that the necessary packages will be available.
This enables the virtual environment.

Exit the virtual environment with
```bash
deactivate
```
