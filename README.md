# Application Processing

## Data

Training data consists of 125000 rows with 122 columns (about 67 Mb), including `ID` and `TARGET`.
The objective is to make predictions for the binary variable called `TARGET`.

The test data consists of 48744 rows with 121 columns (`ID` is included, `TARGET` is omitted).

The input data consists of 120 columns (no `TARGET` and `ID` columns) of which 76 are numerical variables and 44 are categorical.
Even though about two-thirds of the categorical variables contain numbers, they need to be treated as categorical.

Of the 125000 rows in the training data, only 3461 contain zero missing values.
Most rows have between 5 and 49 missing values.
The largest number of missing values in a single row is 61.

### Data cleaning

The data need to be prepared to be used in a model.
This means that categorical variables need to be one-hot encoded and numerical variables may need to be transformed.
There may also be the need to remove certain rows from the data set.

The goal here is to create a transformation on the training set that would allow us to safely fit a model.
This same transformation will also be used on the testing set.

#### Categorical variables

Computational problems with fitting the model may arise if a categorical variable has classes which appear infrequently.
That is, there may be be singularities in our data, possibly leaving us (wrongly) with near zero standard errors.
Accordingly, we may unknowingly underestimate or overestimate our target probability.

Another issue that may arise is if a categorical variable is heavily skewed toward `TARGET == 0` or `TARGET == 1` when the class frequency is too low.
This didn't appear to be a major issue in the training set.

To counter this classes having fewer than 500 occurrences were re-labeled into a new class called `SMALL_GROUP`.
If the combined class of `SMALL_GROUP` is still below 500, then all those classes are set to `NaN`, the place holder for missing values.

From here, indicator variables are created for each class in the variable where a `1` means class presence and `0` means no class presence.
If a class is `NaN`, then a `0` is found for each indicator variable, inidicating no classes.

Summary of findings in the categorical variables.
If a variable is not included, there was no major issue with it:
> `CODE_GENDER`
> - A single instance of `XNA` class, may need to remove this row.
> 
> `NAME_TYPE_SUITE`
> - Some low frequency classes:
>   - `Group of people` ~ 100
>   - `Other_A` ~ 300
> - There are also about 500 `NaN`s, these will be assigned no class
> 
> `NAME_INCOME_TYPE`
> - Several low frequency classes adding up to 19 occurrences, set to `NaN`
> - All 5 unemployed income types received a target value of 1.
> Makes me think that 1 means a rejection.
> 
> `NAME_EDUCATION_TYPE`
> - All 70 `Acedemic degree`s had target 0.
> 
> `NAME_FAMILY_STATUS`
> - A single instance of a class called `Unknown`
> 
> `OCCUPATION_TYPE`
> - Many classes and about 40000 missing
> - Safe to combine low frequency classes into `SMALL_GROUP`
> 
> `ORGANIZATION_TYPE`
> - Similar deal with OCCUPATION_TYPE, but first deal with all
> those "Type X"s. (e.g. there are 13 different industry types, but not
> all have low frequency)
> - Combining like normal, but this could be an area for improvement
> 
> `FONDKAPREMONT_MODE`
> - Many missing values
> 
> `HOUSETYPE_MODE`
> - About half missing
> 
> `WALLSMATERIAL_MODE`
> - About half missing
> 
> `EMERGENCYSTATE_MODE`
> - About half missing

#### Numerical variables

We need to determine which numerical variables are actually categorical.
Meaning they appeared as numbers in the data set but are already just indicator variables.

`FLAG_MOBIL` needs to be removed since all the values are the same.
This variable will not add anything.



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
virtualenv env --python=python3.6   # Creates the environment with a Python 3.6 binary
source env/bin/activate             # Activates the environment
pip install -r requirements.txt     # Installs the packages found in requirements.txt
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
