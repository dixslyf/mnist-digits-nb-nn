#import "@preview/algo:0.3.3": algo, d, i
#import "/constants.typ": *
#import "/lib.typ": *

// SETUP

#set page(
  paper: "a4",
  numbering: "1",
  number-align: right + top,
)

#set heading(numbering: "1.")

#show heading.where(level: 1): it => [
  #set text(size: HEADING_SIZE, weight: "bold")
  #it
]

#show heading.where(level: 2): it => [
  #set text(size: SUBHEADING_SIZE, weight: "regular", style: "italic")
  #it
]

#show heading.where(level: 3): it => [
  #set text(size: SUBSUBHEADING_SIZE, weight: "regular", style: "normal")
  #it
]

#show outline: it => {
  set par(leading: 1em)
  it
}

#set figure(placement: none)

#set block(above: 1.5em)

#set par(justify: true)

#set text(size: TEXT_SIZE, font: TEXT_FONT)

#set list(indent: LIST_INDENT)

#set terms(indent: LIST_INDENT)

#show math.equation: set text(size: 11pt)

#show link: it => {
  set text(fill: blue)
  underline(it)
}

#set list(indent: LIST_INDENT)
#set enum(indent: LIST_INDENT, numbering: "a)")

#show figure: set block(breakable: true)

#show table.cell.where(y: 0): set text(weight: "bold")
#set table(
  stroke: (x, y) => (
    top: if y <= 1 { 0.8pt } else { 0pt },
    bottom: 1pt,
  ),
  inset: 0.5em,
)

// CONTENT

#let date = datetime(
  year: 2024,
  month: 7,
  day: 26,
)

#align(center, [
  #text(size: TITLE_SIZE)[*ICT203 Assignment 2*] \
  \
  #text(size: SUBTITLE_SIZE)[
    Dixon Sean Low Yan Feng \
    Murdoch University \
    #date.display("[day] [month repr:long], [year]")
  ]
])

#outline(indent: auto)

= Problem Description

This report describes the classification of handwritten digits using two different models:
a naive Bayes classifier and a convolutional neural network~(CNN).
The MNIST~(Modified National Institute of Standards and Technology) data set,
widely recognized in machine learning research,
serves as the basis for this analysis.
The data set has been provided as grayscale images of handwritten digits,
each represented as a 28#sym.times\28 pixel NumPy array.
Each image is labeled with a digit from 0 to 9,
corresponding to ten output classes.
The data set has also been pre-partitioned into training, validation, and testing subsets.

= Examining the Given Code and Data

The assignment provides several data and source code files.
The data files are as follows:

- `x_train.npy`

- `y_train.npy`

- `x_val.npy`

- `y_val.npy`

- `x_test.npy`

- `y_test.npy`

Each data file is a binary file containing a NumPy array.
Files prefixed with `x` store image data,
where each array has shape `(n, 28, 28)`,
with `n` representing the number of images.
The second and third dimensions of the shape represent the image's pixels
for a total of 784 (28 by 28) pixels.
Each pixel value ranges from 0 to 255.
Each `x` file has a corresponding `y` file containing the labels
as a one-dimensional array with values ranging from `0` to `9`.

The data set is divided into three pairs of `x` and `y` files:
`train`, `val`, and `test`.
The `train` set is used for model training
while the `test` set is for the final performance evaluation of the models.
The `val` set is supposedly for validation and hyperparameter tuning.

The provided source code files are:

- `main.py`: Entry point providing a command-line interface for training and testing classifiers.

- `naive_bayes.py`: Incomplete implementation of a naive Bayes classifier.
  The `__init__()` function accepts a Laplace smoothing parameter
  to prevent 0 probabilities.

- `alt_model.py:` Incomplete implementation of a PyTorch module for the neural network classifier.

- `nb_data_loader.py`: Contains the `NBDataLoader` class for loading data files and retrieving the train, validation, and test subsets.

- `alt_data_loader.py`: Contains the `ALTDataLoader` class, a subclass of PyTorch's `Dataset` class for reading image data subsets.
  Unfortunately, the name `ALTDataLoader` is a misnomer
  --- PyTorch also provides a `DataLoader` class intended as a wrapper over a `Dataset` to provide batching and shuffling.
  Although `ALTDataLoader` includes `DataLoader` in its name,
  it subclasses `Dataset`, not `DataLoader`.

- `analysis.py`: Loads data subsets, displays their shapes, and visualizes images using `matplotlib`.

= Methodology

== Data Exploration <sec-data-exploration>

The given NumPy arrays have the following shapes:

- `x_train`: `(48000, 28, 28)`

- `y_train`: `(48000,)`

- `x_val`: `(12000, 28, 28)`

- `y_val`: `(12000,)`

- `x_test`: `(10000, 28, 28)`

- `y_test`: `(10000,)`

We observe that there are 48,000 samples in the training set,
12,000 samples in the validation set
and 10,000 samples in the test set
for a total of 70,000 samples.

To begin,
we examine the first 25 samples from the training set
by visualizing them in @fig-image-samples.

#figure(
  caption: [Visualisation of the first 25 samples from the training set],
  image("graphics/image-samples.png")
) <fig-image-samples>

A first observation is that the first image has
a small artefact near the bottom right.
It is not unreasonable to imagine that other images
could have similar noise.
Although the majority of the images do not seem to contain such artefacts,
it is an observation to keep in mind.

A second observation is that most pixels are either almost completely black (`0`) or white (`255`)
--- there are not many pixels that fall in-between.
Pixels whose values fall in-between
appear to be a result of anti-aliasing.
Indeed, if we stack the histograms for each of the 784 pixel values~(@fig-pixel-dists),
we see that the pixels are roughly binary in nature
with a bimodal distribution.

#figure(
  caption: [Stacked histogram for all pixel values],
  image("graphics/pixel-distributions.png", width: 90%)
) <fig-pixel-dists>

Finally, we can surmise that not all pixels are significant for recognising digits.
The first pixel in the top left corner, for example, is black for most,
if not all, images and, hence, does not contribute much to identifying each digit.
We can thus conclude that dimensionality reduction techniques may be useful.

== Choice of Classifiers

=== Naive Bayes <sec-naive-bayes>

There are several variants of naive Bayes classifiers to consider.
For instance, a Bernoulli naive Bayes classifier processes binary input features,
necessitating the conversion of grayscale images to binary images
via, for example, thresholding.
Although feasible, a Bernoulli naive Bayes classifier
(or any other type of naive Bayes classifier for that matter)
assumes independence among input features~(pixels),
which is typically not valid.
Moreover, this method precludes the application of dimensionality reduction techniques
like Principal Component Analysis~(PCA) that do not preserve binary input features.

Instead, a Gaussian naive Bayes classifier was implemented with PCA as a preprocessing step.
Gaussian naive Bayes classifiers assume the continuous input features are normally distributed.
The probability of a specific feature value given a class is estimated by
modeling the feature's distribution with a Gaussian distribution
and computing the probability density.

Despite pixel values not being normally distributed~(@fig-pixel-dists),
PCA transformation yields features that are mostly normally distributed~(@fig-pca-dists).
If these transformed features retain the data set's underlying patterns,
we can expect the Gaussian naive Bayes classifier to perform well.

#figure(
  caption: [Histograms for each value of the training data set projected onto the 1st principal components (using 49 components)],
  image("graphics/pca-distributions.png")
) <fig-pca-dists>

=== Neural Network

For the neural network model,
several architectures were considered,
including multilayer perceptrons~(MLPs),
recurrent neural networks~(RNNs)
and convolutional neural networks~(CNNs).
Given that CNNs utilize convolutional filters to learn local patterns in images,
they are an obvious, optimal choice for classifying handwritten digits.
While MLPs could be employed,
they are likely less effective due to their inability to
exploit the spatial structure of images.
RNNs, suited for sequential data, are not appropriate for this task.

Having selected a CNN,
several architectural decisions remain,
such as the number of layers, convolutional kernel sizes and pooling filter sizes.
These hyperparameters were tuned via hyperparameter tuning~(see @sec-hyperparameter-tuning).
The general architecture of the CNN is outlined as follows:

+ Input images are processed through a series of convolutional layers.
  Each layer's output is passed through an activation function
  and then through a dropout layer for regularization.

+ Following the convolutional layers,
  the data undergoes pooling.
  Typically, pooling occurs after each convolutional layer.
  However, given the relatively small dimensions of 28#sym.times\28 pixels,
  pooling after each convolutional layer would excessively reduce the image size,
  leading to significant information loss or 0-dimensional images.
  Thus, pooling is performed only after all convolutional layers.

+ Post-pooling, the data is flattened and passed through a series of linear layers.
  Similar to the convolutional layers,
  each linear layer's output, except the final layer,
  is passed through an activation function
  and then through a dropout layer for regularization.
  The final linear layer has 10 outputs, each corresponding to a digit (`0`--`9`).
  These outputs represent scores that constitute the model's final prediction.
  Higher scores indicate a higher likelihood that the image corresponds to the respective digit.

== Performance Estimation <sec-performance-estimation>

Cross-validation was performed to estimate the classifiers' performance before final training and testing.
This step is critical for several reasons:

- Providing an unbiased estimate of the classifiers' performance.

- Ensuring that the training and hyperparameter tuning procedures
  provide satisfactory results without overfitting or underfitting the models.

- Determining whether the training and hyperparameter tuning procedures
  need to be amended.

The performance estimation uses nested stratified $k$-fold cross-validation,
with $k = 5$ for both the outer and inner loops.
The inner loop focuses on hyperparameter tuning
(see @sec-hyperparameter-tuning for the procedure),
with only 25 trials due to time constraints.
The outer loop estimates the generalized performance of the models.
Each iteration trains the model on 4 of the 5 folds
and evaluates it on the remaining fold.

The training and validation sets were combined
to form an auxiliary set with 60,000 samples.
The reason for combining them is to have more data for the performance estimation.
It is important to note that combining them does not
leak any validation data to the training procedure
since we are using $k$-fold cross-validation
and not holdout validation,
so each trained model is still evaluated on unseen data.
In theory, performing $k$-fold cross-validation provides a more unbiased estimate of the models' performance.

The performance of the naive Bayes classifier was evaluated by calculating:
(a)~mean accuracy on the outer train folds and
(b)~mean accuracy on the outer validation folds.
For the neural network, additional metrics were used:
(a)~mean loss on the outer train folds and
(b)~mean loss on the outer validation folds.
These metrics help assess if the training procedures result in underfitting or overfitting.

Results for the naive Bayes classifier are as follows:

- Mean train accuracy: 87.4%

- Mean validation accuracy: 87.2%

For the neural network:

- Mean train loss: 0.0128

- Mean train accuracy: 99.7%

- Mean validation loss: 0.0386

- Mean validation accuracy: 98.9%

The similarity in metrics for the train and validation folds suggests that
the models are neither underfitting nor overfitting.
Additionally, the mean accuracies pass the minimum threshold of 80%.
We can expect the final evaluation on the test set to have similar results.

Cross-validation logs are available in the following gzip-compressed files:

- `run_logs/cv_nb.log.gz`

- `run_logs/cv_nn.log.gz`

== Hyperparameter Tuning <sec-hyperparameter-tuning>

Hyperparameter tuning was performed using the #link("https://optuna.org/")[Optuna] library for both the naive Bayes classifier and the CNN.
Instead of using sampling algorithms like grid search (slow)
and random search (suboptimal),
Optuna's default tree-structured Parzen estimator algorithm was used.
As with performance estimation,
the training and validation subsets were combined
to form an auxiliary set (see @sec-performance-estimation for the rationale).

Hyperparameter tuning
was conducted for 100 trials for both models.
Each trial selects a set of parameters
to initialise the model with,
using stratified $k$-fold cross-validation
with $k = 5$
to estimate the model's performance
with that set of parameters.
For the naive Bayes classifier,
hyperparameters were tuned to maximise the mean validation accuracy
over the folds.
For the CNN,
hyperparameters were tuned to minimise the mean validation loss
over the folds.

The following hyperparameters were tuned for the naive Bayes classifier:

- The number of components for principal component analysis~(PCA)

- The type of multiclass classification grouping to use (none, one vs. one, or one vs. rest)

- The Laplace smoothing factor

The following hyperparameters were tuned for the neural network model:

- The batch size

- The number of epochs to train the model for

- The learning rate

- The optimiser algorithm (Adam, Adadelta, stochastic gradient descent or asynchronous stochastic gradient descent)

- The scheduler (exponential or reduction of learning rate on plateau)

  - If using an exponential scheduler, a gamma value is also tuned

  - If reducing the learning rate on plateau, the factor by which the learning rate is reduced is also tuned

- The activation function for each inner layer (ReLU, sigmoid, softplus, SELU or leaky ReLU)

  - If using leaky ReLU, the negative slope is also tuned

- The number of convolutional layers

- The number of output channels for each convolutional layer

- The kernel size for each convolutional layer

- The kernel size for pooling

- The dropout probability for units in a convolutional layer

- The dropout probability for units in a linear layer

- The number of output features for each linear layer (except the last, whose number of outputs is fixed to 10)

The best parameters for the naive Bayes classifier achieved a mean validation accuracy of 87.3%:

- Number of PCA components: 67

- Multiclass classification grouping: none

- Laplace smoothing factor: 0.23584673798488343

The best parameters for the neural network achieved a mean validation loss of 0.026529311973722845:

- Batch size: 64

- Number of epochs: 15

- Learning rate: 0.845073619862434

- Optimiser algorithm: Adadelta

- Scheduler: exponential

  - Gamma: 0.7041331729249224

- Activation function: ReLU

- Number of convolutional layers: 3

  - Number of output channels for layer 0: 41

  - Kernel size for layer 0: 3

  - Number of output channels for layer 1: 123

  - Kernel size for layer 1: 3

  - Number of output channels for layer 2: 119

  - Kernel size for layer 2: 4

- Kernel size for pooling: 3

- Dropout probability for units in a convolutional layer: 0.27438816607290456

- Number of linear layers: 2

  - Number of output features for linear layer 0: 126

  - Number of output features for linear layer 1: 10 (fixed)

- Dropout probability for units in a linear layer: 0.23955112992546726

The following files are the Optuna journal files, which keep track of information for each trial:

- `nb_journal.log`

- `nn_journal.log`

Logs (compressed with gzip) for hyperparameter tuning can be found in:

 - `tune_nb.log.gz`

 - `tune_nn.log.gz`

== Training and Testing <sec-training-testing>

The Naive Bayes classifier and CNN were trained using the optimal hyperparameters identified through tuning.
Training was conducted on the training subset (`x_train.npy` and `y_train.npy`).
Post-training, the models were serialized and saved to disk:
the naive Bayes classifier to `nb.pkl` and the CNN to `nn.pt`.
Training of the naive Bayes classifier was performed without
acceleration from a graphical process unit~(GPU) and took 2.03 seconds.
The CNN was trained with GPU acceleration via CUDA and took 93.52 seconds.
This significantly longer training time compared to the naive Bayes classifier is expected
since CNNs are much more complex and have much more parameters to tune.
Logs for training can be found in `train_nb.log.gz` and `train_nn.log.gz`.

For the final evaluation, the models were deserialized and evaluated on the test set.
In addition to generating a confusion matrix,
the following metrics were calculated:

- Accuracy

- Precision

- Recall

- F1-score

- Support

A confusion matrix is a table that shows the counts of
true positive, true negative, false positive, and false negative predictions
for each class.
Each row represents the actual class,
while each column represents the predicted class,
helping to visualize where the model correctly or incorrectly classified each instance.

Precision measures the percentage of correctly classified samples for a class
out of the total number of samples predicted as that class.
For example, a precision of 80% for class `0` means that,
out of the samples predicted `0`,
80% were actually `0`.
On the other hand, recall measures the percentage of correctly classified samples for a class
out of the total number of samples that truly belong to that class.
For example, a recall of 80% for class `0` means that,
out of the truly `0` samples,
80% were predicted correctly.
F1-score is calculated as $frac(2 dot.c "precision" dot.c "recall", "precision" + "recall")$
and represents the harmonic mean of the precision and recall.
We can hence interpret the F1-score as an average of the precision and recall.
Support is the number of true samples of a class in the test data set.

Precision, recall, F1-score and support were computed for each class.
Additionally, macro and weighted averages of these metrics were determined.
A macro averages calculates a metric independently for each class
and then takes the mean, treating all classes equally.
A weighted average accounts for the number of instances in each class when averaging the metrics,
balancing the influence of each class based on its support.
A micro average aggregates the contributions of all classes
by considering the total true positives, false positives, and false negatives across all classes.
However, no micro averages were calculated as they are equivalent to accuracy when considering all classes.

The naive Bayes classifier took 0.36 seconds (again, without GPU acceleration) to make predictions for the test set
whereas the CNN took 3.35 seconds (with GPU acceleration via CUDA).
The longer prediction time for the CNN is expected
since CNNs have many more calculations to compute
due to their more complex architecture
compared to the naive Bayes classifier,
despite using GPU acceleration
The raw output metrics for testing can be found in `test_nb.json` and `test_nn.json`.

= Evaluation Results and Discussion

#let table-scores(scores) = table(
  columns: 5,
  align: (col, row) => if col == 0 {
    horizon + right
  } else {
    horizon + center
  },
  table.header(
    [],
    ..scores
      .at("0")
      .keys()
      .map(h => upper(h.at(0)) + h.slice(1))
    ),
  ..scores
    .pairs()
    .map(((key, scores)) => if key == "accuracy" {
      (
        none, none, none, none, none,
        key, none, none, scores, none,
      )
    } else {
      (key, ..scores.values())
    })
    .flatten()
    .map(v => if type(v) == float and v >= 0 and v <= 1.0 { [#calc.round(v * 100, digits: 1)\%] } else { v }) // Round floats.
    .map(v => if type(v) == str { upper(v.at(0)) + v.slice(1) } else { v }) // Capitalise the first letter of strings.
    .map(v => if type(v) == str { v.replace(regex("avg"), "average") } else { v }) // Replace "avg" with "average".
    .map(v => if type(v) == str and v.replace(regex("[0-9]"), "") == "" { "Class " + v } else { v }) // Append "class" to digit labels.
    .map(v => [#v])
)

== Naive Bayes

The evaluation scores and confusion matrix for the naive Bayes classifier
are shown in @fig-nb-scores and @fig-nb-confusion respectively.
We observe an accuracy of 88.2%,
which is similar to the estimated accuracy from @sec-performance-estimation.
Additionally, the macro averages for precision, recall and F1-score
are also 88.2%.
The weighted averages are similar to the macro averages
--- this is expected since the distribution of the digits are close to uniform.
For a classifier that assumes conditional independence of the input features,
this is a decent result.

Classes `0` and `1` have the highest precision and recall (and, hence, F1-score),
indicating that the classifer performs well on these digits.
Indeed, looking at the confusion matrix,
we see relatively few misclassifications

On the other hand, the rest of the digits have varying scores.
In particular, the model only has an F1-score of 82.4% for class `5`.
Looking at the precision and recall for `5`,
we see a precision of 80% and a recall of 85%,
meaning that
85% of the actual `5`s were correctly identified,
and, when the model predicts a digit as `5`,
it is correct 80% of the time.
From the confusion matrix,
we see that the model tends to incorrectly predict a `5` as a `3`,
and sometimes as an `8`.
The converse is also true --- it tends to incorrectly predict `3`s and `8`s as `5`s.
In fact, the model also appears to incorrectly predict `3`s as `8`s
and vice versa.
We can conclude that the classifier struggles to distinguish between
`5`, `3` and `8`.

Interestingly, class `7` has a relatively high precision of 91.9%,
but a recall of only 84.4%.
That is, there are relatively few false positives
but a high number of false negatives for `7`.
Looking at the confusion matrix,
`7` tends to be misclassified as a `9` or `2`.
It also gets misclassified as `4` and `1` to a lesser extent.
Similarly, images incorrectly predicted as `7` are usually actually a `9` or `2`,
but the extent of this is not as great as misclassifying `7` images,
agreeing with the precision and recall scores.

Overall, based on the F1-score for each class,
the order of the digits the model performs from best to worst
is: `1`, `0`, `6`, `7`, `4`, `3`, `9`, `2`, `8`, `5`.

#let nb-scores = json("data/test_nb.json")
#[
#show table.cell.where(x: 0): set text(style: "italic")
#show figure: set block(breakable: false)
#figure(
  caption: [Evaluation scores for the naive Bayes classifier],
  table-scores(nb-scores),
) <fig-nb-scores>
]

#figure(
  caption: [Confusion matrix for the naive Bayes classifier],
  image("graphics/confusion-matrix-nb.png"),
) <fig-nb-confusion>

== Neural Network

The CNN achieved a mean loss of 0.02130503880043543.
The evaluation scores and confusion matrix for the CNN
are shown in @fig-nb-scores and @fig-nb-confusion respectively.
The CNN performed with a high accuracy of 99.3%,
which is similar to the estimated accuracy from @sec-performance-estimation.
Additionally, the model also performed with 99.3% for the macro and weighted averages
of precision, recall and F1-score.
As explained in the results for the naive Bayes model,
the weighted and macro averages are expected to be similar
since the distribution of the digits are close to uniform.
It is clear that the CNN outperforms the naive Bayes classifier.

Looking at the precision and recall for the individual classes,
we see that they are all around 99%.
In other words, whenever the CNN predicts a digit,
it is almost always right;
and nearly every digit is classified correctly.
Indeed, if we look at the confusion matrix,
there are extremely little misclassifications.
Out of the digits, `9` has the highest number of misclassifications~(16 misclassifications),
which agrees with the fact that it has the lowest F1-score.
In total, there are 66 misclassifications out of the 10,000 images.

Overall, based on the F1-score for each class,
the order of the digits the model performs from best to worst
is: `1`, `2`, `0`, `4`, `3`, `8`, `6`, `7`, `5`, `9`.

#let nn-scores = json("data/test_nn.json")
#[
#show table.cell.where(x: 0): set text(style: "italic")
#show figure: set block(breakable: false)
#figure(
  caption: [Evaluation scores for the neural network],
  table-scores(nn-scores),
)
]

#figure(
  caption: [Confusion matrix for the neural network],
  image("graphics/confusion-matrix-nn.png"),
)

= Discussion

It is clear that the CNN greatly outperforms the naive Bayes classifier
---
every metric calculated for the CNN is higher than those for the naive Bayes classifier.
This result is expected for two reasons:

+ CNNs are highly suitable for image data.

+ Naive Bayes classifiers assume conditional independence of the input features given the class,
  which is almost certainly not the case for the pixels of the handwritten digit images.

It is surprising, however, that the naive Bayes classifier could still achieve
decent scores
despite its assumptions,
considering the pixels of the images should be highly correlated.

Although the methodology used in this experiment is comprehensive
and follows the typical procedure for machine learning,
there are several improvements that can have been made.
For example, we could have tried other dimensionality reduction techniques
besides PCA,
such as feature selection and t-distributed stochastic neighbor embedding~(t-SNE),
for the Gaussian naive Bayes classifier.
Furthermore, we could also have compared the performance of a Bernoulli naive Bayes classifier
against the Gaussian naive Bayes classifier.
The selection of these techniques can be achieved through hyperparameter tuning,
and can potentially lead to better results for the naive Bayes classifier.

For the CNN,
although many hyperparameters were tuned,
the overall structure of the CNN is still rather rigid.
For example, dropout is always applied for regularisation
even though other similar techniques exist,
such as batch normalisation.
Although an extensive set of hyperparameters was considered,
there are still many architectural decisions left unexplored,
such as the use of vision transformers instead of CNNs.

= User Guide

== Running Locally

*Note:* I recommend running the program on Kaggle instead~(see @sec-running-on-kaggle)
to take advantage of their graphical processing units~(GPUs).
Setting up CUDA or ROCm for a local machine (if you have a compatible GPU)
is, unfortunately, finicky when using Poetry on Windows.

Ensure you have the following requirements:

- Python `^3.11`

- Poetry `^1.7.1`

If the program does not work on Python `3.12`,
fall back to `3.11` as that was the version the program was tested on locally.
To install Poetry, follow the instructions #link("https://python-poetry.org/docs/#installation")[here].

To run the program,
run the following in your shell
in the root directory of the project (i.e., the directory containing `pyproject.toml`):

```bash
  $ poetry install
  $ poetry run python -m assignment
```

Alternatively, you may run `poetry shell` after `poetry install`
to avoid having to type `poetry run`:

```bash
  $ poetry install
  $ poetry shell
  $ python -m assignment
```

It is also possible to run the program without using `poetry`.
However, you would have to manually install the packages
listed in `pyproject.toml` using `pip` or your package manager of choice.
Alternatively, you may want to try an Anaconda distribution
as it supposedly sets up GPU acceleration automatically;
however, this is not guaranteed to work since the package versions may differ
from the ones used in this project.

== Running on Kaggle <sec-running-on-kaggle>

It is also possible to run the program on Kaggle.
In fact, the longer stages of the methodology
were performed on Kaggle.
Follow the following steps:

#[
#set enum(indent: LIST_INDENT, numbering: "1.")

1. Create and name a new _dataset_, using the assignment zip file as the source.
   We'll refer to this dataset as the `ict202-assignment-2` dataset.
   The trick here is that we are importing the entire set of code and data files as a dataset
   so that we can manually execute the code using shell commands
   from a Kaggle notebook.

2. Create a new notebook.

3. In the notebook, add the `ict202-assignment-2` dataset as an input (under the _Input_ section in the pane on the right).

4. Create a new code cell in the notebook that executes a shell command using the `!` prefix.
   The cell should first use `cd` to change the directory to the directory of the `ict202-assignment-2` dataset input.
   You can retrieve the directory of the `ict202-assignment-2` dataset input
   by hovering over its entry in the _Input_ section of the right pane
   and clicking the copy icon.

   Then, using `&&` to chain another command,
   run `python -m assignment <subcommand> <options>`,
   replacing `<subcommand>` with a subcommand
   and `<options>` with a list of command line options.

   If you are performing training (which will save the model),
   due to file permissions on Kaggle,
   you must set the output file path to one with `/kaggle/working/` as a prefix.

   As an example, if you want to train the neural network model,
   the cell should contain contents similar to the following:

   ```
   !cd "/kaggle/input/ict202-assignment-2" && python -m assignment train nn -o "/kaggle/working/nn.pt"
   ```

  "`/kaggle/input/ict202-assignment-2`" should be replaced with the
  path of the `ict202-assignment-2` dataset that was copied earlier.
]

Note, however, that the steps above do not use the package versions pinned by Poetry.
Instead, they use the packages installed on Kaggle's machines.
Although the program will most likely run fine,
there is no 100% guarantee that it will (the pain of Python packaging).

== Usage

The interface of the program is significantly different from the provided code.
First, there is only one executable program.
The program expects the first argument to be one of the following subcommands:

- `analyse`: Performs data exploration tasks~(corresponds to @sec-data-exploration).

- `cv`: Performs cross-validation~(corresponds to @sec-performance-estimation).

- `tune`: Performs hyperparameter tuning~(corresponds to @sec-hyperparameter-tuning).

- `train`: Performs model training~(corresponds to @sec-training-testing).

- `test`: Performs model testing~(corresponds to @sec-training-testing).

Note that optional flags (i.e., those starting with a `-`) *must* be specified after the last positional argument
(positional arguments are required arguments that do not start with "`-`").
For example, do _not_ do this:

```bash
  $ python -m assignment -o "/kaggle/working/nn.pt" train nn
```

Instead, do:

```bash
  $ python -m assignment train nn -o "/kaggle/working/nn.pt"
```

Each subcommand has its own set of argument and options
that you can view using the `-h` or `--help` flag.

There are 2 optional global flags:

- `-d --data-dir`: Specifies the directory to read the data set from. Defaults to `data/digitdata`.

- `-r --random-state`: Specifies the random state to use for stochastic processes to have deterministic output. Defaults to `0`.

=== `analyse`

`analyse` takes one positional argument specifying what type of analysis to perform:

- `shapes`: View the shape of each provided NumPy array.

- `samples`: Visualize the first 25 images.

- `pixel-distributions`: Visualize the distribution of each pixel value in a histogram (@fig-pixel-dists).

- `pca-distributions`:  Visualize the distribution of each PCA component when using 49 components (@fig-pca-dists).

Examples:

```bash
  $ python -m assignment analyse shapes
  $ python -m assignment analyse pca-distributions
```

=== `cv`

The `cv` subcommand estimates the performance of each model as described in @sec-performance-estimation.
`cv` expects one positional argument specifing which model to estimate the performance of: `nb` or `nn`.
The following optional flags can be specified:

- `-o OUTER_FOLDS, --outer-folds OUTER_FOLDS`: the number of outer folds (default: 5) to use for nested cross-validation.

- `-i INNER_FOLDS, --inner-folds INNER_FOLDS`: the number of inner folds (default: 5) to use for nested cross-validation.

- `-t TRIALS, --trials TRIALS`: the number of Optuna trials for hyperparameter tuning (default: 25)

- `-j JOBS, --jobs JOBS`: the number of parallel jobs to run on processors (default: 1)

Additionally, when `nn` is specified, the following optional flag can also be set:

- `-g {auto,cpu,cuda,mps}, --device {auto,cpu,cuda,mps}`: the compute device to use for neural network operations (default: `auto`).

Note that `cv` will take a long time to run (several hours)
with the default parameters.
You can specify a lower number of outer folds, inner folds and trials to
speed up the process,
at the expense of much less accurate approximations of the performance.

If more than 1 job is used,
note that the results will not be deterministic.

*Warning:* I do not recommend lowering the number of trials below 25
because it is possible for Optuna to fail.
Optuna has been configured to perform trial pruning,
by which trials producing a hyperparameter set that
does not appear to perform well
(e.g., based on the non-convergence of loss)
are skipped.
If the number of trials is too low,
it is possible for all trials to be pruned,
leading to no completed trials and an error.

Examples:

```bash
  $ python -m assignment cv nb -o 3 -i 3 -j 2
  $ python -m assignment cv nn -o 3 -i 3 -g cpu
```

=== `tune`

`tune` performs hyperparameter tuning.
Like `cv`, the first positional argument specifies the model whose hyperparameters to tune: `nb` or `nn`.
The following optional flags can be specified:
- `-m {tune,view-best,view-all}, --mode {tune,view-best,view-all}`:
  the mode to run in (`tune`, `view-best` or `view-all`) (default: `view-best`)

- `-t TRIALS, --trials TRIALS`:
  the number of trials for hyperparameter tuning (default: 25)

- `-f FOLDS, --folds FOLDS`:
  the number of folds for $k$-fold cross-validation (default: 5)

- `-j JOBS, --jobs JOBS`: the number of parallel jobs (default: 1)

- `-l JOURNAL_PATH, --journal-path JOURNAL_PATH`:
  the path to the Optuna journal to load and view tuned parameters (default: `nb_journal.log` for `nb` and `nn_journal.log` for `nn`)

- `-n STUDY_NAME, --study-name STUDY_NAME`:
  the name of the Optuna study to load from the journal (default: `nb-study` for `nb` and `nn-study` for `nn`)

As before, if `nn` is specified, the compute device can be set:

- `-g {auto,cpu,cuda,mps}, --device {auto,cpu,cuda,mps}`:
  the device to use for neural network operations (default: `auto`)

Examples:

```bash
  # Tune the hyperparameters for the naive Bayes classifier
  # with 50 trials and 10 folds.
  # Note that this will append the results to
  # the journal and study if they exist.
  $ python -m assignment tune nb -m tune -t 50 -f 10

  # Fetch the best hyperparameters for the naive Bayes classifier.
  # You can only fetch after the hyperparameters have been tuned
  # (e.g., through the previous commmand).
  $ python -m assignment tune nb

  # Fetch all tested hyperparameters for the naive Bayes classifier.
  # You can only fetch after the hyperparameters have been tuned.
  $ python -m assignment tune nb -m view-all

```

=== `train`

`train` performs training on the training set and saves the trained model to a file.
The first positional argument specifies the model to train: `nb` or `nn`.
The following optional flags can be specified:

- `-o OUTPUT_PATH, --output-path OUTPUT_PATH:`
  the file path to save the trained model to (default: `nb.pkl` for `nb` and `nn.pt` for `nn`)

If training the neural network (`nn`), the following additional flags can be specified:

- `-g {auto,cpu,cuda,mps}, --device {auto,cpu,cuda,mps}`:
  the device to use for neural network operations (default: `auto`)

- `-b BATCH_SIZE, --batch-size BATCH_SIZE`:
  override for the number of samples in each batch (default: None)

- `-e EPOCHS, --epochs EPOCHS`:
  override for the number of epochs to train the model for (default: None)

- `-l LEARNING_RATE, --learning-rate LEARNING_RATE`:
  override for the learning rate (default: None)

By default, the batch size, number of epochs and learning rate used
are those determined through hyperparameter tuning.
However, you can override them using their respective override options.

=== `test`

The `test` subcommand evaluates a trained model on the test set.
As with most other subcommands,
the first positional argument specifies the model to evaluate.
The following optional flags can be specified:

- `-i INPUT_PATH, --input-path INPUT_PATH`:
  the file path to load the trained model from (default: `nb.pkl` for `nb` and `nn.pt` for `nn`)

If testing the neural network (`nn`),
the following option is also available:

- `-g {auto,cpu,cuda,mps}, --device {auto,cpu,cuda,mps}`:
  the device to use for neural network operations (default: `auto`)

Results are printed to the standard output stream.
A figure is plotted for the confusion matrix using `matplotlib`.
