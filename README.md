# Phase Transitions in the Output Distribution of Large Language Models
## Julian Arnold, Flemming Holtorf, Frank Schäfer, Niels Lörch

This repository contains code to run $g$-dissimilarity scans as presented in the related article.

## Abstract of the article

In a physical system, changing parameters such as temperature can induce a phase transition: an abrupt change from one state of matter to another. Analogous phenomena have recently been observed in large language models. Typically, the task of identifying phase transitions requires human analysis and some prior understanding of the system to narrow down which low-dimensional properties to monitor and analyze. Statistical methods for the automated detection of phase transitions from data have recently been proposed within the physics community. These methods are largely system agnostic and, as shown here, can be adapted to study the behavior of large language models. In particular, we quantify distributional changes in the generated output via statistical distances, which can be efficiently estimated with access to the probability distribution over next-tokens. This versatile approach is capable of discovering new phases of behavior and unexplored transitions -- an ability that is particularly exciting in light of the rapid development of language models and their emergent capabilities.

## How to use

* `my_models.py` provides wrappers to load language models.
* `geneval.py` contains functions to generate outputs from a language model and evaluate their probabilities.
* `confusion.py` contains functions to calculate $g$-dissimilarities from these probabilities.

the notebooks load these `.py` files to demonstrate simple scans of the prompt, temperature, and training epoch.
