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

= Examining the Provided Code and Data

Several source code and data files were provided by the assignment.
We have the following data files:

- `x_train.npy`

- `y_train.npy`

- `x_val.npy`

- `y_val.npy`

- `x_test.npy`

- `y_test.npy`

Each file is a binary file storing a NumPy array.
Files with the `x` prefix contain image data for the handwritten digits.
Each `x` array has the following shape: `(n, 28, 28)`,
where `n` is the number of images in the array —
`n` differs for `x_train`, `x_val` and `x_test`
as they each have a different number of images.

Each `x` file has a corresponding `y` file
containing the labels for the images.
For example, `y_train` contains the image labels for `x_train`.
Each `y` array is one-dimensional and contains the same number of
values as there are images in its corresponding `x` array.
The values range from `0` to `9` to represent the digits.

There are three different `x`-`y` pairs of files: `train`, `val` and `test`.
The `train` pair is meant to be fed to the classifiers for _training_.
On the other hand, the `val` pair is for _validation_ —
that is, for estimating the generalisation performance of the classifiers.
The validation set can also be used as part of hyperparameter tuning.
Finally, the `test` pair is for evaluating the performance of the final trained classifiers.

We have the following source code files:

- `main.py`: Provides a commandline interface that lets the user train and test the classifiers with various parameters.

- `naive_bayes.py`: Provides an incomplete implementation of a naive Bayes classifier.

- `alt_model.py`: Provides an incomplete implementation of a PyTorch module.

- `nb_data_loader.py`: Provides the `NBDataLoader` class, which loads the data files described before and allows callers to retrieve the train, validation and test subsets of the image data.

- `alt_data_loader.py`: Provides the `ALTDataLoader` class, a subclass of PyTorch's `Dataset` that reads either the train, validation or test image data depending on what the user specifies.
  Unfortunately, the class is poorly named —
  PyTorch provides the `DataLoader` and `Dataset` classes for handling data.
  Although `ALTDataLoader` is a `Dataset`, it is _not_ a `DataLoader`.

- `analysis.py`: Loads the various data subsets, displays their shapes and visualises one of the images with `matplotlib`.

= Methodology

== Data Exploration

The given NumPy arrays have the following shapes:

- `x_train`: `(48000, 28, 28)`

- `y_train`: `(48000,)`

- `x_val`: `(12000, 28, 28)`

- `y_val`: `(12000,)`

- `x_test`: `(10000, 28, 28)`

- `y_test`: `(10000,)`

That is, we have 48,000 samples in our train set,
12,000 samples in our validation set
and 10,000 samples in our test set
for a total of 70,000 samples.

To get a feel of what the data looks like,
the first 25 samples from the train set were visualised
and are shown in @fig-image-samples.

#figure(
  caption: [Visualisation of the first 25 samples from the train set],
  image("graphics/image-samples.png")
) <fig-image-samples>

A first observation we can make is that the first image has
a small artefact near the bottom right.
It is not unreasonable to imagine that other images
could have similar noisy artefacts.
However, the majority do not seem to contain such artefacts.

A second observation is that most pixels are either completely black (`0`) or completely white (`255`).
It appears that pixels whose values fall in-between
are a result of anti-aliasing.
Indeed, if we plot a histogram for each of the 784 pixel values (@fig-pixel-dists),
we see that the pixels are roughly binary in nature
with a bimodal distribution.

#figure(
  caption: [Stacked histogram for all pixel values],
  image("graphics/pixel-distributions.png", width: 90%)
) <fig-pixel-dists>

Finally, we can surmise that not all pixels are significant for recognising digits.
The first pixel in the top left corner, for example, is black for most,
if not all, images and, hence, does not contribute much to the identifying each digit.
Ergo, dimensionality reduction techniques will be useful.
Specifically, principal component analysis (PCA) was applied
to the data set for the naive Bayes classifier (see @sec-naive-bayes).

== Choice of Classifiers

=== Naive Bayes <sec-naive-bayes>

There are several variants of naive Bayes classifiers to choose from.
For example, we can use a Bernoulli naive Bayes classifier,
which handles binary input features —
we would have to convert the grayscale images to binary images
through, for example, thresholding.
While this is a viable approach,
a Bernoulli naive Bayes classifier
(or any other naive Bayes classifier for that matter)
assumes that the input features (in this case, the pixel values) are independent,
which is almost certainly not the case.
Moreover, we lose the ability to apply dimensionality reduction techniques
that would not preserve the binary nature of the input features
(e.g., principal component analysis).

Instead, a Gaussian naive Bayes classifier was implemented,
with principal component analysis (PCA)
as a preprocessing step.
Gaussian naive Bayes classifiers assume their (continuous) input features
are normally distributed.
The probability of a specific input feature value given a class
can then be estimated
by approximating the input feature's distribution (conditional on the class)
with a Gaussian distribution 
and calculating the probability density for the value.
From @fig-pixel-dists, however, we know that the pixel values
are _not_ normally distributed, and, hence,
we can expect that a Gaussian naive Bayes classifier would likely perform poorly.
If, however, we apply PCA to the data set,
we observe that the resulting features are mostly normally distributed (@fig-pca-dists).
Assuming the resulting features retain the underlying patterns of the data set,
we can expect a Gaussian naive Bayes classifier to perform well.

#figure(
  caption: [Histograms for each value of the data set projected onto the 1st principal components (using 49 components)],
  image("graphics/pca-distributions.png")
) <fig-pca-dists>


= Evaluation Results and Discussion

= User Guide
