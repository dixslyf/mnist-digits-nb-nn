# MNIST Digit Classification: Naive Bayes vs. CNN

This project evaluates and compares two models for handwritten digit classification on the MNIST dataset:
a Gaussian Naive Bayes classifier and a Convolutional Neural Network (CNN).
It was developed as part of a university assignment.

The code provides a command-line interface written in Python
for training, testing, cross-validation and hyperparameter tuning (with [Optuna](https://optuna.readthedocs.io/)).
The Naive Bayes model uses [scikit-learn](https://scikit-learn.org/) with principal component analysis (PCA) as a preprocessing step
while the CNN is implemented with [PyTorch](https://pytorch.org/).

As with most university assignments,
a report was required.
You can read the full report from the latest release
[here](https://github.com/dixslyf/mnist-digits-nb-nn/releases/latest/download/report.pdf) (PDF).

## Results

| Model       | Accuracy | Precision | Recall | F1 Score |
| ----------- | -------- | --------- | ------ | -------- |
| Naive Bayes | 88.2%    | 88.4%     | 88.2%  | 88.3%    |
| CNN         | 99.3%    | 99.3%     | 99.3%  | 99.3%    |

The numbers above are the _weighted averages_ of their values for all classes.

The CNN significantly outperforms the Naive Bayes classifier, as expected for image classification tasks.
However, Naive Bayes demonstrates reasonable performance considering its simplicity and underlying assumptions.

Logs and the final trained models are available from the [latest release](https://github.com/dixslyf/mnist-digits-nb-nn/releases/latest).

## Running

First, ensure you have the following installed:

- Python 3.10+

- Poetry 1.7.1+

Then, run the following commands:

```sh
poetry install
poetry run python -m mnist_digits_nb_nn <subcommand>
```

Alternatively, to avoid typing `poetry` repeatedly,
you can first activate the shell:

```sh
poetry shell
python -m mnist_digits_nb_nn <subcommand>
```

### Nix

This project provides a Nix flake,
which you can use to build and run the command-line tool.
To build, run the following from the repository's root directory:

```
nix build .
```

To run the tool:

```
nix run . -- <args>
```

Alternatively, you can perform the above without having a local clone of the repository:

```
nix build github:dixslyf/mnist-digits-nb-nn
nix run github:dixslyf/mnist-digits-nb-nn -- <args>
```

**Note:** You may need to set the `MPLBACKEND` environment variable to `TkAgg` for plots to show properly
when running the `analyse` subcommand.

It is also recommended that you use the developer shell provided by the flake for CUDA to work properly.

## Usage

The command-line tool expects a subcommand as the first argument.
To view help for any subcommand, run:

```sh
# Assuming you have activated the venv shell
python -m mnist_digits_nb_nn <subcommand> --help
```

### Available Subcommands

| Subcommand | Description                                                         |
| ---------- | ------------------------------------------------------------------- |
| `analyse`  | Performs basic data analysis and visualisations                     |
| `cv`       | Runs nested cross-validation to estimate generalisation performance |
| `tune`     | Performs hyperparameter tuning using Optuna                         |
| `train`    | Trains a model using provided or tuned parameters                   |
| `test`     | Evaluates a trained model on the test set                           |

### Examples

Train the CNN and save the model:

```bash
python -m mnist_digits_nb_nn train nn -o models/nn.pt
```

Test a previously saved Naive Bayes model:

```bash
python -m mnist_digits_nb_nn test nb -i models/nb.pkl
```

Perform CNN hyperparameter tuning with 50 trials:

```bash
python -m mnist_digits_nb_nn tune nn -m tune -t 50
```

## Compiling the Report

The report is written with [Typst](https://typst.app/).

To compile the report, ensure you have the Typst compiler installed.
Then, run the following in the project's root directory:

```sh
typst compile report/report.typ
```

This will output a PDF document at `report/report.pdf`.

### Nix

The report can also be compiled through the project's Nix flake.
To do so, run the following from the repository's root directory:

```
nix build .#report
```

Alternatively, you can also compile the report without having a local clone:

```
nix build github:dixslyf/mnist-digits-nb-nn#report
```

The resulting PDF document can then be accessed through the usual `result` symlink.
