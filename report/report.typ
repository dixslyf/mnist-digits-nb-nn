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

= Choice of the Alternative Classifier

= Methodology

= Evaluation Results and Discussion

= User Guide
