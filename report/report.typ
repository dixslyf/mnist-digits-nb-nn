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

= Choice of the Alternative Classifier

= Methodology

= Evaluation Results and Discussion

= User Guide
