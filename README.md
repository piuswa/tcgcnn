# Comparing Different CNN Architectures to Classify Trading Cards

This repository contains the code for a paper handed in for the Scientific Writing course at the University of Basel in the fall semester of 2025.

## Running the code

This code depends on this dataset: https://www.kaggle.com/datasets/archanghosh/yugioh-database/versions/1

The dataset needs to be in the same directory as the code when the code is executed. ``preprocessing.py`` needs to be run first as it transforms the data. Afterwards to models can be trained with or without using oversampling by executing ``train_all_oversample.py`` or ``train_all_normal.py``. 