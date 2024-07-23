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

- `alt_model.py:` Incomplete implementation of a PyTorch module for the neural network classifier.

- `nb_data_loader.py`: Contains the `NBDataLoader` class for loading data files and retrieving the train, validation, and test subsets.

- `alt_data_loader.py`: Contains the `ALTDataLoader` class, a subclass of PyTorch's `Dataset` class for reading image data subsets.
  Unfortunately, the name `ALTDataLoader` is a misnomer
  --- PyTorch also provides a `DataLoader` class intended as a wrapper over a `Dataset` to provide batching and shuffling.
  Although `ALTDataLoader` includes `DataLoader` in its name,
  it subclasses `Dataset`, not `DataLoader`.

- `analysis.py`: Loads data subsets, displays their shapes, and visualizes images using `matplotlib`.

= Methodology

== Data Exploration

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

=== Hyperparameter Tuning <sec-hyperparameter-tuning>

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

== Training and Testing

The Naive Bayes classifier and CNN were trained using the optimal hyperparameters identified through tuning.
Training was conducted on the training subset (`x_train.npy` and `y_train.npy`).
Post-training, the models were serialized and saved to disk:
the naive Bayes classifier to `nb.pkl` and the CNN to `nn.pt`.
Training of the naive Bayes classifier was performed without
acceleration from a graphical process unit~(GPU) and took 2.03 seconds.
The CNN was trained with GPU acceleration via CUDA and took 93.52 seconds.
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
