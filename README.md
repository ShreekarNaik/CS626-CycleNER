# CS626 Course Project

This Project is an implementation of CycleNER an [Original Paper](https://dl.acm.org/doi/10.1145/3485447.3512012) by Andrea Iovine, Anjie Fang, Besnik Fetahu, Oleg Rokhlenko, Shervin Malmasi.

## Quick Overview
CycleNER uses cycle-consistency training for two functions: sequence-to-entity (S2E); and entity-to-sequence (E2S), to learn NER tags from some seed set of sentences and another set of entity examples. The output from one function is the input the the other and the algorithm attempts to align each representation space, thus, learning to tag named entities in an unsupervised manner.
